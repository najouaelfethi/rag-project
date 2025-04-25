#PDF parsing logic (text, tables, images, etc)
import fitz
import re
from typing import List, Dict, Any
from loguru import logger
from dataclasses import dataclass

#hierarchical PDF structure
@dataclass #storing data in structured way
class PDFSection:
    title: str                      # The heading title
    level: int                      # Heading level, like 0(top level=>documemt), 1 (big title: main section), 2 (sub title)
    content: List[str]             # List of paragraphs/text under this section
    subsections: List['PDFSection']  # Subsections inside this section (like a nested tree)
    page_num: int                   # Page number where the section starts
    metadata: Dict[str, Any]        # Extra info, like font size, bounding box, etc


class PDFExtractor:
    def __init__(self):#constructor
        self.patterns = { #identify headings, citations, references, etc
            'citation': r'\[\s*\d+\s*\]',  # Matches [1], [2] in the text
            'reference': r'\([A-Za-z]+\s*et\s*al\.\s*\d{4}\)',  # Matches (Author et al. 2024)
            'heading': r'^[A-Z][A-Za-z\s]+$',  # Matches lines like "INTRODUCTION"
            'list_item': r'^\s*[\d\.\)]\s+',  # Matches 1. or a) or •
        }

        # sets font sizes that will help guess if a line is a title or subtitle. Bigger = more important.
        self.heading_sizes = {
            'h1': 16, 'h2': 14, 'h3': 12
        }
    
    def extract(self, file_path: str) -> PDFSection:
        try:
            #open pdf
            doc = fitz.open(file_path)  
            root_section = PDFSection("Document", 0, [], [], 0, {})#all will be updated
            current_section = root_section

            #loop through each page
            for page_num, page in enumerate(doc):
                #text_dict={'blocks': {'type':0,bbox:[] lines:['bbox':[], 'spans':[{'text':'','size': ,'flags': ,'font':'','color': ,'origin':[] }]]} , {'type':1, 'bbox':[],'image':True}} 
                #type=0 is text block, type=1 is image block, type=2 is drawing bloc
                text_dict = page.get_text("dict")  # Extracts both text AND formatting
                
                for block in text_dict["blocks"]:#loop through each block
                    if block["type"] == 0:  
                        text = "" #collect text from current block
                        font_sizes = set() #collect unique font sizes of block including titles,subtitles,... everything: {...,...,...,}

                        for line in block["lines"]:#loop through block line by line
                            for span in line["spans"]:#loop through each line's span, span is a piece of text with same font style(size,..)
                                text += span["text"]#
                                font_sizes.add(span["size"]) 

                        if not text.strip():#skip empty blocks
                            continue
                        
                        max_font = max(font_sizes)
                        if max_font >= self.heading_sizes['h1'] and re.match(self.patterns['heading'], text):
                            section = PDFSection(
                                title=text.strip(),
                                level=1,#big title
                                content=[],
                                subsections=[],
                                page_num=page_num,
                                metadata={
                                    'font_size': max_font,
                                    'bbox': block["bbox"]
                                }
                            )
                            root_section.subsections.append(section)
                            current_section = section
                        
                        elif max_font >= self.heading_sizes['h2'] and re.match(self.patterns['heading'], text):
                            section = PDFSection(
                                title=text.strip(),
                                level=2,#sub title
                                content=[],
                                subsections=[],
                                page_num=page_num,
                                metadata={
                                    'font_size': max_font,
                                    'bbox': block["bbox"] #It’s a rectangle that defines where the block of text appears on the page
                                }
                            )
                            current_section.subsections.append(section)
                        else:
                            # Add to current section
                            current_section.content.append(text.strip())
            
            return root_section
                    
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise






