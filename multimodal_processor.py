import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch
from torchvision import transforms
from PIL import Image
import re

class MultimodalProcessor:
    def __init__(self, model_embeddings):
        self.model_embeddings = model_embeddings
        self.image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.ocr_processor = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")
        
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF files including text, images, tables, and mathematical expressions"""
        docs = []
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            if text.strip():
                docs.append(Document(page_content=text, metadata={"page": page_num, "type": "text"}))
            
            # Extract images
            images = page.get_images()
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_doc = self._process_image(image_bytes)
                if image_doc:
                    docs.append(image_doc)
            
            # Extract tables
            tables = page.find_tables()
            for table in tables:
                table_doc = self._process_table(table)
                if table_doc:
                    docs.append(table_doc)
            
            # Extract mathematical expressions
            math_expressions = self._extract_math_expressions(text)
            for expr in math_expressions:
                docs.append(Document(page_content=expr, metadata={"page": page_num, "type": "math"}))
        
        return docs
    
    def _process_image(self, image_bytes: bytes) -> Document:
        """Process images using OCR and image captioning"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # OCR processing
            ocr_text = pytesseract.image_to_string(image)
            
            # Image captioning
            image_tensor = self.image_processor(image).unsqueeze(0)
            caption = self.ocr_processor(image_tensor)
            
            # Combine OCR and caption
            combined_text = f"OCR Text: {ocr_text}\nImage Description: {caption}"
            
            return Document(
                page_content=combined_text,
                metadata={"type": "image"}
            )
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def _process_table(self, table: Any) -> Document:
        """Process tables and convert to structured text"""
        try:
            # Extract table data
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            
            # Convert to pandas DataFrame for better text representation
            df = pd.DataFrame(table_data)
            table_text = df.to_string(index=False)
            
            return Document(
                page_content=f"Table Content:\n{table_text}",
                metadata={"type": "table"}
            )
        except Exception as e:
            print(f"Error processing table: {e}")
            return None
    
    def _extract_math_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text"""
        # Regular expression pattern for mathematical expressions
        math_pattern = r'\$\$.*?\$\$|\$.*?\$|\\[.*?\\]|\\\(.*?\\\)'
        expressions = re.findall(math_pattern, text)
        return expressions
    
    def process_chart(self, image_bytes: bytes) -> Document:
        """Process charts and diagrams"""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract text using OCR
            chart_text = pytesseract.image_to_string(img)
            
            # Generate description
            description = f"Chart with {len(contours)} elements. Text content: {chart_text}"
            
            return Document(
                page_content=description,
                metadata={"type": "chart"}
            )
        except Exception as e:
            print(f"Error processing chart: {e}")
            return None
    
    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chunk documents while preserving multimodal content structure"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunked_docs = []
        for doc in documents:
            if doc.metadata["type"] in ["text", "math"]:
                # Split text and math content
                chunks = text_splitter.split_documents([doc])
                chunked_docs.extend(chunks)
            else:
                # Keep images, tables, and charts as single chunks
                chunked_docs.append(doc)
        
        return chunked_docs 