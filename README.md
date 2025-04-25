# RAG Pipeline

A robust pipeline for processing and embedding multi-modal documents for Retrieval-Augmented Generation (RAG) systems.

## Features

- Multi-modal document processing (text, images, tables, diagrams)
- Chunking and embedding generation
- Vector storage and metadata management
- Support for various document formats (PDF, DOCX, PPTX, XLSX)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the pipeline settings in `config.py`

2. Process a document:
```python
from pipeline_orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
result = orchestrator.process_file("path/to/your/document.pdf")
```

3. Access the processed content:
```python
# Get JSON representation
json_data = orchestrator.to_json(result)

# Access individual content items
for item in result.content_items:
    print(f"Type: {item.type}")
    print(f"Content: {item.content[:100]}...")
    print(f"Embedding shape: {item.embedding.shape}")
```

## Pipeline Stages

1. **Ingestion**: File loading and format detection
2. **Parsing**: Content extraction from documents
3. **Modality Detection**: Identifying content types
4. **Chunking**: Breaking content into manageable pieces
5. **Feature Extraction**: Processing different content types
6. **Embedding**: Generating vector representations
7. **Storage**: Saving vectors and metadata

## Configuration

The pipeline can be configured through `config.py`:

- `PIPELINE_CONFIG`: General pipeline settings
- `MODEL_CONFIG`: Model specifications
- `STORAGE_CONFIG`: Storage backend settings

## License

MIT License 