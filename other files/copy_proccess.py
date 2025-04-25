from transformers import pipeline
from PIL import Image
import fitz  # PyMuPDF
import torch
from typing import List, Dict, Any
import numpy as np
from loguru import logger
from models.models_config import MODEL_CONFIGS

class MultimodalProcessor:
    def __init__(self):
        self.models = {
            name: pipeline(config["task"], model=config["model_name"])
            for name, config in MODEL_CONFIGS.items()
        }

    def process_document(self, file_path: str) -> Dict[str, Any]:
        try:
            # 1. Document Loading
            doc = self._load_document(file_path)
            
            # 2. Content Extraction
            content = self._extract_content(doc)
            
            # 3. Multimodal Processing
            processed_content = self._process_content(content)
            
            return processed_content
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _load_document(self, file_path: str):
        """Load document based on type."""
        if file_path.endswith('.pdf'):
            return fitz.open(file_path)
        
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _extract_content(self, doc) -> Dict[str, Any]:
        content = {
            "text_blocks": [],
            "images": [],
            "tables": [],
            "other": []
        }
        
        if isinstance(doc, fitz.Document):
            # PDF processing
            for page in doc:
                # Extract text
                text = page.get_text("dict")
                content["text_blocks"].extend(text["blocks"])
                
                # Extract images
                images = page.get_images()
                content["images"].extend(images)
                
                # Extract tables
                tables = page.find_tables()
                content["tables"].extend(tables)
        
        return content

    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        processed = {
            "text": [],
            "images": [],
            "tables": [],
            "charts": [],
            "diagrams": [],
            "math": []
        }
        
        # Process text blocks
        for block in content["text_blocks"]:
            if "text" in block:
                if self._is_math(block["text"]):
                    processed["math"].append(self._process_math(block))
                else:
                    processed["text"].append(self._process_text(block))
        
        # Process images
        for img in content["images"]:
            img_type = self._classify_image(img)
            if img_type == "chart":
                processed["charts"].append(self._process_chart(img))
            elif img_type == "diagram":
                processed["diagrams"].append(self._process_diagram(img))
            else:
                processed["images"].append(self._process_image(img))
        
        return processed

    def _is_math(self, text: str) -> bool:
        """Detect if text contains math formulas."""
        result = self.models["math_detector"](text)
        return result[0]["label"] == "MATH"

    def _classify_image(self, image) -> str:
        # First try chart detection
        chart_result = self.models["chart_detector"](image)
        if chart_result[0]["label"] == "chart":
            return "chart"
        
        # Then check for diagram patterns
        objects = self.models["object_detector"](image)
        if self._is_diagram(objects):
            return "diagram"
        
        return "regular"

    def _process_image(self, image) -> Dict[str, Any]:
        #regular images
        # Get image classification
        classification = self.models["image_classifier"](image)
        
        # Extract text from image
        ocr_text = self.models["ocr"](image)
        
        # Get visual features
        objects = self.models["object_detector"](image)
        
        return {
            "type": "image",
            "classification": classification[0],
            "ocr_text": ocr_text[0]["generated_text"],
            "objects": objects
        }

    def _process_chart(self, image) -> Dict[str, Any]:
        """Process charts with specialized analysis."""
        # Get chart type and data
        chart_type = self.models["chart_detector"](image)
        
        # Extract data using VQA
        data = self._extract_chart_data(image)
        
        return {
            "type": "chart",
            "chart_type": chart_type[0],
            "data": data
        }

    def _extract_chart_data(self, image) -> Dict[str, Any]:
        """Extract data from charts using VQA."""
        questions = [
            "What is the title of this chart?",
            "What are the axis labels?",
            "What are the data values?"
        ]
        
        data = {}
        for q in questions:
            result = self.models["vqa"](image=image, question=q)
            data[q] = result[0]["answer"]
        
        return data