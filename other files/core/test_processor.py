import os
import pytest
from loguru import logger
from core.processor import MultimodalProcessor
import fitz  # PyMuPDF

TEST_PDF = "./documents/attention_pdf.pdf"#Agent-white-paper-google.pdf

# This creates a processor object that we'll use in all tests
@pytest.fixture
def processor():#processor object
    return MultimodalProcessor()

# Test 1: Check if processor is created correctly => PASSED
def test_processor_initialization(processor):
    assert processor is not None #assert checks if something is true
    assert hasattr(processor, 'models')
    assert isinstance(processor.models, dict)
"""
# Test 2: Try to load a PDF file 
def test_load_pdf(processor):
    # Try to load the PDF
    pdf_document = processor._load_documents(TEST_PDF)
    
    # Check if loading was successful
    assert pdf_document is not None
    assert isinstance(pdf_document, fitz.Document)
    
    # Print how many pages are in the PDF
    print(f"\nPDF loaded successfully with {len(pdf_document)} pages")

# Test 3: Extract content from PDF
def test_extract_content(processor):
    # Load the PDF
    pdf_document = processor._load_documents(TEST_PDF)
    
    # Extract content
    content = processor._extract_content(pdf_document)
    
    # Check if we got the expected content types
    assert "text_blocks" in content
    assert "images" in content
    assert "tables" in content
    
    # Print what we found
    print(f"\nFound {len(content['text_blocks'])} text blocks")
    print(f"Found {len(content['images'])} images")
    print(f"Found {len(content['tables'])} tables")
    
    # text blocks content
    print("\n=== Text Blocks Content ===")
    for i, block in enumerate(content["text_blocks"][:2]):  # Show first 5 blocks
        if "text" in block:
            print(f"\nText Block {i+1}:")
            print(f"Content: {block['text']}")

# Test 4: Process the entire document
def test_process_document(processor):
    # Process the document
    result = processor.process_document(TEST_PDF)
    
    # Check if we got all the expected results
    assert "text" in result
    assert "math" in result
    assert "images" in result
    assert "tables" in result
    
    # Print what we found
    print("\n=== Document Processing Results ===")
    print(f"Text blocks: {len(result['text'])}")
    print(f"Math formulas: {len(result['math'])}")
    print(f"Images: {len(result['images'])}")
    print(f"Tables: {len(result['tables'])}")
    
    # text content after processing
    print("\n=== Text Content ===")
    for i, text_block in enumerate(result["text"][:2]):  # Show first 5 blocks
        print(f"\nText Block {i+1}:")
        print(f"Content: {text_block.get('text', '')}")
    
    # math formulas after processing
    print("\n=== Math Formulas ===")
    for i, math_block in enumerate(result["math"][:5]):
        print(f"\nMath Formula {i+1}:")
        print(f"Content: {math_block.get('text', '')}")

# Test 5: Check error handling
def test_error_handling(processor):
    with pytest.raises(Exception):
        processor.process_document("file_that_does_not_exist.pdf")
"""

if __name__ == "__main__":
    # Run all tests with output capture disabled
    pytest.main(["-v", "-s", "core/test_processor.py"])

# text blocks: 194
# images: 2
# tables: 3
# maths: 5-7
