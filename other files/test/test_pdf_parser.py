import sys
import os
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)#telling python to go see for modules to import

from parsers.pdf_parser import PDFExtractor, PDFSection
from pprint import pprint
import pytest


def print_section(section, indent=0, max_sections=5, count=[0]):
    if count[0] >= max_sections:
        return

    prefix = "  " * indent
    print(f"{prefix} Title: {section.title} (Level {section.level})")
    print(f"{prefix} Page number: {section.page_num}")
    if section.content:
        print(f"{prefix} Content: {section.content[:1]}...")  # print first paragraph only

    count[0] += 1

    for subsection in section.subsections:
        if count[0] >= max_sections:
            break
        print_section(subsection, indent + 1, max_sections, count)

#Get all content from a section
def get_all_content(section):
    content = section.content.copy()
    for subsection in section.subsections:
        content.extend(get_all_content(subsection))
    return content

#Test 1: Basic structure
def test1_pdf_extraction():
    file_path = "./documents/attention_pdf.pdf"        
    extractor = PDFExtractor()
    sections = extractor.extract(file_path)

    #print extracted structure
    print("\nExtracted PDF Structure:")
    print_section(sections)

    #check root section if it's OK
    assert sections.title == "Document"
    assert sections.level == 0

    #check if it has real sections
    assert len(sections.subsections) > 0

    #check content if it's empty or not
    all_content = get_all_content(sections)
    all_text = " ".join(all_content)
    assert len(all_text) > 0

    #check for each section title,level,content,subsections
    for section in sections.subsections:
        assert section.title
        assert section.level > 0
        assert isinstance(section.content, list)
        assert isinstance(section.subsections, list)


#Test 2: Look for citations
def test2_pdf_extraction_with_citations():
    file_path = "./documents/attention_pdf.pdf"        
    extractor = PDFExtractor()
    sections = extractor.extract(file_path)
    
    # Get all content including subsections
    all_content = get_all_content(sections)
    all_text = " ".join(all_content)
    
    # Go through each word in the text and pick the ones that looks like [1], [2],..
    citations = [text for text in all_text.split() if text.startswith("[") and text.endswith("]")]
    if citations:
        print("\nFound citations:", citations)
        
    if citations: #make sure at least one of the citations looks valid
        assert any(citation.startswith("[") and citation.endswith("]") for citation in citations)

#Test 3: Look for headings
def test3_pdf_extraction_with_headings():
    file_path = "./documents/attention_pdf.pdf"        
    extractor = PDFExtractor()
    sections = extractor.extract(file_path)

    #check if there is any headings
    print("\nFound headings:")
    for section in sections.subsections:
        print(f"Level {section.level}: {section.title}")
    
    #checking if there is at least one section
    assert len(sections.subsection) > 0
    #validate content of each section
    for section in sections.subsections:
        assert section.title
        assert section.level > 0
        assert isinstance(section.content, list)#is content stored as list?


if __name__ == "__main__":
    test1_pdf_extraction()
    #test2_pdf_extraction_with_citations()
    #test3_pdf_extraction_with_headings()