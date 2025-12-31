"""
ChromaDB Document Storage and Retrieval System
This program extracts documents from the data folder, chunks them,
stores them in ChromaDB, and provides a CLI for querying.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from sentence_transformers import SentenceTransformer
import argparse
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
load_dotenv()
model_name = os.getenv("model_name") or "all-MiniLM-L6-v2"  # Default model if not set

class ChromaDBManager:
    """Manages ChromaDB operations including document storage and retrieval."""
    
    def __init__(self, model_name: str = None, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client and collection."""
        # Use provided model_name or fallback to default
        if model_name is None:
            model_name = os.getenv("model_name") or "all-MiniLM-L6-v2"
        
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = "Lov_data_legal_documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}...")
        try:
            self.embedding_model = SentenceTransformer(model_name)
            print("Embedding model loaded successfully!")
        except ValueError as e:
            error_msg = str(e)
            if "modernbert" in error_msg.lower() or "does not recognize this architecture" in error_msg.lower():
                print(f"\n⚠️  Error: The model '{model_name}' uses an unsupported architecture.")
                print("This may be due to:")
                print("  1. The model requires a newer version of transformers library")
                print("  2. The model architecture is not yet supported")
                print(f"\nFalling back to default model: 'all-MiniLM-L6-v2'")
                print("To fix this, either:")
                print("  - Update transformers: pip install --upgrade transformers")
                print("  - Use a different model in your .env file")
                print("  - Recommended models: all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L6-v2\n")
                # Fallback to default model
                model_name = "all-MiniLM-L6-v2"
                self.embedding_model = SentenceTransformer(model_name)
                print(f"Successfully loaded fallback model: {model_name}")
            else:
                print(f"Error loading embedding model: {error_msg}")
                raise
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            print(f"\nTrying fallback model: 'all-MiniLM-L6-v2'")
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Successfully loaded fallback model!")
            except Exception as fallback_error:
                print(f"Fallback model also failed: {str(fallback_error)}")
                raise ValueError(f"Could not load any embedding model. Original error: {str(e)}")
        
    def load_xml_file(self, file_path: Path) -> Dict:
        """Load and extract text content from XML file."""
        try:
            tree = ET.parse(str(file_path))
            root = tree.getroot()
            
            # Extract all text from XML elements
            def extract_text(element):
                """Recursively extract text from XML elements."""
                text_parts = []
                if element.text and element.text.strip():
                    text_parts.append(element.text.strip())
                for child in element:
                    text_parts.append(extract_text(child))
                    if child.tail and child.tail.strip():
                        text_parts.append(child.tail.strip())
                return ' '.join(text_parts)
            
            content = extract_text(root)
            xml_name = file_path.stem  # Filename without extension
            
            return {
                'content': content,
                'source': str(file_path),
                'metadata': {
                    'xml_name': xml_name,
                    'file_name': file_path.name,
                    'file_type': 'xml'
                }
            }
        except ET.ParseError as e:
            raise Exception(f"XML parsing error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error reading XML file: {str(e)}")
    
    def load_documents(self, data_folder: str) -> List[Dict]:
        """Load all supported documents from the data folder."""
        documents = []
        data_path = Path(data_folder)
        
        if not data_path.exists():
            print(f"Error: Data folder '{data_folder}' does not exist!")
            return documents
        
        supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
        }
        
        for file_path in data_path.rglob('*'):
            if file_path.is_file():
                file_ext = file_path.suffix.lower()
                
                # Handle XML files separately
                if file_ext == '.xml':
                    try:
                        xml_doc = self.load_xml_file(file_path)
                        documents.append(xml_doc)
                        print(f"Loaded: {file_path.name} (XML name: {xml_doc['metadata']['xml_name']})")
                    except Exception as e:
                        print(f"Error loading {file_path.name}: {str(e)}")
                
                # Handle other file types
                elif file_ext in supported_extensions:
                    try:
                        loader_class = supported_extensions[file_ext]
                        loader = loader_class(str(file_path))
                        loaded_docs = loader.load()
                        
                        for doc in loaded_docs:
                            # Extract filename for consistency
                            file_name = Path(file_path).stem
                            doc_metadata = doc.metadata.copy()
                            doc_metadata['file_name'] = Path(file_path).name
                            
                            documents.append({
                                'content': doc.page_content,
                                'source': str(file_path),
                                'metadata': doc_metadata
                            })
                        print(f"Loaded: {file_path.name}")
                    except Exception as e:
                        print(f"Error loading {file_path.name}: {str(e)}")
        
        return documents
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List[Dict]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = []
        for doc in documents:
            text_chunks = text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'content': chunk,
                    'source': doc['source'],
                    'chunk_index': i,
                    'metadata': doc['metadata']
                })
        
        return chunks
    
    def store_chunks(self, chunks: List[Dict]):
        """Store document chunks in ChromaDB."""
        if not chunks:
            print("No chunks to store!")
            return
        
        print(f"Storing {len(chunks)} chunks in ChromaDB...")
        
        # Clear existing collection if it has data
        existing_count = self.collection.count()
        if existing_count > 0:
            print(f"Clearing existing {existing_count} documents...")
            # Get all existing IDs and delete them
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        
        # Prepare data for batch insertion
        ids = []
        contents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"chunk_{idx}_{Path(chunk['source']).stem}"
            ids.append(chunk_id)
            contents.append(chunk['content'])
            metadatas.append({
                'source': chunk['source'],
                'chunk_index': chunk['chunk_index'],
                **chunk.get('metadata', {})
            })
            
        # Generate embeddings and store
        embeddings = self.embedding_model.encode(contents).tolist()
        
        # Store in batches to handle large datasets
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_contents,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            print(f"Stored batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
        
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")
    
    def query(self, query_text: str, n_results: int = 3) -> List[Dict]:
        """Query ChromaDB and return top N results."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()[0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results


def setup_and_store(data_folder: str = "./data", chunk_size: int = 1000, 
                   chunk_overlap: int = 200):
    """Setup ChromaDB and store documents from data folder."""
    print("=" * 60)
    print("ChromaDB Setup and Document Storage")
    print("=" * 60)
    
    # Initialize ChromaDB manager
    db_manager = ChromaDBManager(model_name=model_name)
    
    # Load documents
    print(f"\nLoading documents from '{data_folder}'...")
    documents = db_manager.load_documents(data_folder)
    
    if not documents:
        print("No documents found in the data folder!")
        print("Supported formats: .txt, .pdf, .docx, .xml")
        return None
    
    print(f"Loaded {len(documents)} document(s)")
    
    # Chunk documents
    print(f"\nChunking documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = db_manager.chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"Created {len(chunks)} chunk(s)")
    
    # Store chunks
    print("\nStoring chunks in ChromaDB...")
    db_manager.store_chunks(chunks)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    
    return db_manager


def query_interface(db_manager: ChromaDBManager = None):
    """Interactive command line interface for querying."""
    if db_manager is None:
        print("Loading ChromaDB...")
        db_manager = ChromaDBManager(model_name=model_name)
    
    print("\n" + "=" * 60)
    print("ChromaDB Query Interface")
    print("=" * 60)
    print("Enter your queries (type 'exit' or 'quit' to stop)")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\nSearching for: '{query}'")
            print("-" * 60)
            
            results = db_manager.query(query, n_results=3)
            
            if not results:
                print("No results found.")
            else:
                print(f"\nTop {len(results)} results:\n")
                for i, result in enumerate(results, 1):
                    print(f"[Result {i}]")
                    # Display XML name if available, otherwise show source
                    if 'xml_name' in result['metadata']:
                        print(f"XML Name: {result['metadata']['xml_name']}")
                    print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                    if result['distance'] is not None:
                        print(f"Similarity Distance: {result['distance']:.4f}")
                    print(f"Content:\n{result['content']}")
                    print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="ChromaDB Document Storage and Retrieval System"
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup ChromaDB and store documents from data folder'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Query the database (returns top 3 results)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive query interface'
    )
    parser.add_argument(
        '--data-folder',
        type=str,
        default='./data',
        help='Path to data folder (default: ./data)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for document splitting (default: 1000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Chunk overlap size (default: 200)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    db_manager = None
    
    # Setup mode
    if args.setup:
        db_manager = setup_and_store(
            data_folder=args.data_folder,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
    
    # Single query mode
    if args.query:
        if db_manager is None:
            db_manager = ChromaDBManager(model_name=model_name)
        
        print(f"\nQuery: {args.query}")
        print("=" * 60)
        results = db_manager.query(args.query, n_results=3)
        
        if not results:
            print("No results found.")
        else:
            print(f"\nTop {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"[Result {i}]")
                print(f"Source: {result['metadata'].get('source', 'Unknown')}")
                if result['distance'] is not None:
                    print(f"Similarity Distance: {result['distance']:.4f}")
                print(f"Content:\n{result['content']}")
                print("-" * 60)
    
    # Interactive mode
    if args.interactive:
        query_interface(db_manager)


if __name__ == "__main__":
    main()

