# legalComplianceChromaDB
A Codebase that converts documents into vectors and stores them in ChromaDB for semantic search and retrieval.

## Features
- Extract documents from the `data` folder (supports .txt, .pdf, .docx, .xml)
- Automatically chunk documents for efficient storage
- Store document chunks in ChromaDB with embeddings
- XML files: XML filename is stored as a key in metadata (`xml_name`)
- Query interface to retrieve top 3 most relevant chunks
- Command-line interface for easy interaction

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation

### 1. Install Dependencies

Install all required Python packages:

```bash
pip install -r chormaDBFunc/requirement.txt
```

### 2. Environment Configuration (Optional)

Create a `.env` file in the project root to configure the embedding model:

```bash
# .env file
model_name=all-MiniLM-L6-v2
```

**Recommended Models:**
- `all-MiniLM-L6-v2` (default, fast and efficient)
- `all-mpnet-base-v2` (better accuracy, slower)
- `paraphrase-MiniLM-L6-v2` (good for semantic similarity)

If no `.env` file is provided, the system will use `all-MiniLM-L6-v2` as the default model.

## Database Setup

### Step 1: Prepare Your Data

Place your documents in the `data` folder. The system supports:

- **XML files** (`.xml`) - Recommended for structured legal documents
- Text files (`.txt`)
- PDF documents (`.pdf`)
- Word documents (`.docx`)

**Folder Structure:**
```
legalComplianceChromaDB/
├── data/
│   ├── document1.xml
│   ├── document2.xml
│   ├── document3.pdf
│   └── ...
```

### Step 2: XML File Placement and Structure

#### XML File Placement
1. Create or navigate to the `data` folder in the project root
2. Place all your XML files directly in the `data` folder or in subdirectories
3. The system will recursively search for all XML files in the `data` folder

#### XML File Structure
Your XML files can have any structure. The system will extract all text content from all XML elements. Example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<legal_document>
    <header>
        <title>Compliance Policy</title>
    </header>
    <content>
        <section>
            <title>Introduction</title>
            <paragraph>This document outlines...</paragraph>
        </section>
    </content>
</legal_document>
```

**Important Notes:**
- XML filename (without extension) is stored as `xml_name` key in metadata
- For example: `compliance_policy.xml` → `xml_name: "compliance_policy"`
- All text content from XML elements is extracted and chunked
- XML structure is preserved in the extracted text

### Step 3: Initialize ChromaDB and Store Documents

Run the setup command to:
1. Load all documents from the `data` folder
2. Extract text content (for XML: extracts all element text)
3. Split documents into chunks
4. Generate embeddings using the configured model
5. Store everything in ChromaDB

```bash
python chormaDBFunc/main.py --setup
```

**What Happens During Setup:**
1. **Document Loading**: All supported files in `data/` folder are loaded
   - XML files: Text extracted from all XML elements
   - Other formats: Content extracted using appropriate loaders
2. **Chunking**: Documents are split into chunks (default: 1000 characters with 200 character overlap)
3. **Embedding Generation**: Each chunk is converted to a vector embedding
4. **Storage**: Chunks are stored in ChromaDB collection: `Lov_data_legal_documents`
5. **Metadata Storage**: Each chunk includes:
   - `source`: Full file path
   - `xml_name`: XML filename (for XML files only)
   - `file_name`: Full filename
   - `chunk_index`: Chunk position in document

**Example Output:**
```
============================================================
ChromaDB Setup and Document Storage
============================================================

Loading embedding model: all-MiniLM-L6-v2...
Embedding model loaded successfully!

Loading documents from './data'...
Loaded: compliance_policy.xml (XML name: compliance_policy)
Loaded: regulations.pdf
Created 15 chunk(s)

Storing chunks in ChromaDB...
Stored batch 1/1
Successfully stored 15 chunks in ChromaDB!

============================================================
Setup complete!
============================================================
```

### Step 4: Verify Database Setup

The ChromaDB database is automatically created in the `chroma_db/` folder. You should see:
- `chroma_db/chroma.sqlite3` - SQLite database file
- Database persists between sessions

## Query Matching and Retrieval

### How Query Matching Works

1. **Query Processing**: Your query text is converted to an embedding vector using the same model
2. **Similarity Search**: ChromaDB searches for chunks with the most similar embeddings
3. **Top Results**: Returns the top 3 most relevant chunks based on cosine similarity
4. **Result Display**: Shows chunk content, source file, XML name (if applicable), and similarity score

### Query Methods

#### Method 1: Single Query

Query the database with a single search term:

```bash
python chormaDBFunc/main.py --query "data protection regulations"
```

**Example Output:**
```
Query: data protection regulations
============================================================

Top 3 results:

[Result 1]
XML Name: compliance_policy
Source: data/compliance_policy.xml
Similarity Distance: 0.2341
Content:
Organizations must implement robust data protection measures including
encryption, access controls, and regular security audits. Personal data
must be handled in accordance with GDPR and other applicable regulations.
------------------------------------------------------------
```

#### Method 2: Interactive Query Mode

Start an interactive session for multiple queries:

```bash
python chormaDBFunc/main.py --interactive
```

**Interactive Session:**
```
============================================================
ChromaDB Query Interface
============================================================
Enter your queries (type 'exit' or 'quit' to stop)
------------------------------------------------------------

Query: employment law compliance
Searching for: 'employment law compliance'
------------------------------------------------------------

Top 3 results:
[Result 1]
XML Name: hr_policies
Source: data/hr_policies.xml
...

Query: exit
Goodbye!
```

#### Method 3: Setup and Query in One Command

Setup database and immediately start querying:

```bash
python chormaDBFunc/main.py --setup --interactive
```

### Query Tips

1. **Natural Language**: Use natural language queries - the system understands semantic meaning
   - ✅ Good: "What are the data protection requirements?"
   - ✅ Good: "employment law compliance"
   - ✅ Good: "GDPR regulations"

2. **Specific Terms**: More specific queries yield better results
   - ✅ Better: "data encryption requirements for personal information"
   - ❌ Less specific: "data"

3. **XML Name Filtering**: Results show `XML Name` when the chunk comes from an XML file, making it easy to identify the source document

## Advanced Configuration

### Custom Data Folder

Specify a different folder for your documents:

```bash
python chormaDBFunc/main.py --setup --data-folder ./my_documents
```

### Chunk Size Configuration

Adjust chunk size and overlap for different document types:

```bash
# Smaller chunks for detailed documents
python chormaDBFunc/main.py --setup --chunk-size 500 --chunk-overlap 100

# Larger chunks for comprehensive context
python chormaDBFunc/main.py --setup --chunk-size 2000 --chunk-overlap 400
```

**Chunk Size Guidelines:**
- **Small (500-800)**: Better for precise matching, more chunks
- **Medium (1000-1500)**: Balanced (default recommended)
- **Large (2000+)**: Better context, fewer chunks

### All Options

```bash
python chormaDBFunc/main.py --setup \
    --data-folder ./custom_data \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --interactive
```

## Supported Document Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| XML | `.xml` | XML filename stored as `xml_name` key in metadata |
| Text | `.txt` | Plain text files |
| PDF | `.pdf` | PDF documents |
| Word | `.docx` | Microsoft Word documents |

## Project Structure

```
legalComplianceChromaDB/
├── chormaDBFunc/
│   ├── main.py              # Main program
│   ├── requirement.txt      # Dependencies
│   └── init.py
├── data/                    # Place your documents here
│   ├── document1.xml        # XML files (recommended)
│   ├── document2.xml
│   └── ...
├── chroma_db/               # ChromaDB storage (auto-created)
│   └── chroma.sqlite3       # Database file
├── .env                     # Environment config (optional)
└── README.md
```

## Troubleshooting

### Model Loading Errors

If you see errors about unsupported model architectures (e.g., `modernbert`):

1. **Update transformers library:**
   ```bash
   pip install --upgrade transformers
   ```

2. **Use a different model** in your `.env` file:
   ```
   model_name=all-MiniLM-L6-v2
   ```

3. **Let it auto-fallback**: The system will automatically use `all-MiniLM-L6-v2` if your specified model fails

### No Results Found

- Ensure you've run `--setup` first
- Check that documents are in the `data` folder
- Verify file formats are supported (.xml, .txt, .pdf, .docx)
- Try more specific or different query terms

### XML Files Not Loading

- Verify XML files are well-formed (valid XML syntax)
- Check file permissions
- Ensure files have `.xml` extension
- Review error messages in console output

### Database Issues

- Delete `chroma_db/` folder and run `--setup` again to recreate
- Ensure write permissions in project directory
- Check disk space availability

## Example Workflow

1. **Install dependencies:**
   ```bash
   pip install -r chormaDBFunc/requirement.txt
   ```

2. **Place XML files in data folder:**
   ```bash
   # Copy your XML files
   cp /path/to/your/*.xml ./data/
   ```

3. **Setup database:**
   ```bash
   python chormaDBFunc/main.py --setup
   ```

4. **Query documents:**
   ```bash
   python chormaDBFunc/main.py --query "your search term"
   ```

   Or use interactive mode:
   ```bash
   python chormaDBFunc/main.py --interactive
   ```

## Notes

- The database persists between sessions - you only need to run `--setup` when adding new documents
- To update with new documents, simply add them to the `data` folder and run `--setup` again (existing data will be cleared and replaced)
- XML filenames are automatically used as keys in metadata for easy identification
- Query matching uses semantic similarity, not just keyword matching
