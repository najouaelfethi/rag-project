from transformers import pipeline
from PIL import Image
import fitz  # PyMuPDF
import torch
from typing import List, Dict, Any
import numpy as np
from loguru import logger
from models.models_config import MODEL_CONFIGS

model = OpenAIEmbeddings(openai_api_key=openai_key, openai_organization=openai_org)

file_path = "./documents/attention_pdf.pdf"
class MultimodalProcessor:
    def __init__(self):
        # Only initialize text-related models
        self.models = {
            "text_processor": pipeline("text-classification", model="distilbert-base-uncased"),
            "math_detector": pipeline("text-classification", model="distilbert-base-uncased")
        }

    def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            # 1. Document Loading
            doc = self._load_document(file_path)
            
            # 2. Content Extraction
            content = self._extract_content(doc)
            
            # 3. Text Processing
            processed_content = self._process_content(content)
            
            return processed_content
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _load_document(self, file_path: str):
        if file_path.endswith('.pdf'):
            return fitz.open(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _extract_content(self, doc) -> Dict[str, Any]:
        content = {
            "text_blocks": [],
            "tables": [],
            "metadata": {}
        }
        
        if isinstance(doc, fitz.Document):
            # PDF processing
            for page in doc:
                # Extract text with layout information
                text_dict = page.get_text("dict")
                content["text_blocks"].extend(text_dict["blocks"])
                
                # Extract tables
                tables = page.find_tables()
                content["tables"].extend(tables)
                
                # Extract metadata
                content["metadata"].update({
                    "page_count": len(doc),
                    "page_size": page.rect,
                    "rotation": page.rotation
                })
        
        return content

    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        processed = {
            "text": [],
            "tables": [],
            "math": [],
            "metadata": content["metadata"]
        }
        
        # Process text blocks
        for block in content["text_blocks"]:
            if "text" in block:
                text = block["text"].strip()
                if not text:
                    continue
                    
                # Check if it's math content
                if self._is_math(text):
                    processed["math"].append({
                        "text": text,
                        "type": "math",
                        "bbox": block.get("bbox", [])
                    })
                else:
                    # Process regular text
                    processed["text"].append({
                        "text": text,
                        "type": "text",
                        "bbox": block.get("bbox", []),
                        "font_size": block.get("font_size", 0),
                        "font_name": block.get("font_name", "")
                    })
        
        # Process tables
        for table in content["tables"]:
            processed["tables"].append({
                "data": table.extract(),
                "bbox": table.bbox
            })
        
        return processed

    def _is_math(self, text: str) -> bool:
        """Simple math detection based on common math symbols."""
        math_symbols = ['=', '+', '-', '*', '/', '^', '√', '∫', '∑', '∏', '∞']
        return any(symbol in text for symbol in math_symbols)

    def get_document_structure(self, processed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document structure based on text formatting."""
        structure = {
            "sections": [],
            "tables": processed_content["tables"],
            "math_equations": processed_content["math"]
        }
        
        # Group text by font size to identify headings
        text_blocks = processed_content["text"]
        font_sizes = {}
        
        for block in text_blocks:
            size = block["font_size"]
            if size not in font_sizes:
                font_sizes[size] = []
            font_sizes[size].append(block)
        
        # Identify headings based on font size
        heading_sizes = sorted(font_sizes.keys(), reverse=True)[:3]  # Top 3 font sizes
        
        # Create sections
        current_section = None
        for size in heading_sizes:
            for block in font_sizes[size]:
                if size == heading_sizes[0]:  # Main heading
                    current_section = {
                        "title": block["text"],
                        "level": 1,
                        "content": [],
                        "subsections": []
                    }
                    structure["sections"].append(current_section)
                elif size == heading_sizes[1]:  # Subheading
                    if current_section:
                        subsection = {
                            "title": block["text"],
                            "level": 2,
                            "content": []
                        }
                        current_section["subsections"].append(subsection)
                else:  # Regular text
                    if current_section:
                        current_section["content"].append(block["text"])
        
        return structure
