from processors.image_processor import ImageTextProcessor  

if __name__ == "__main__":
    file_path = "./documents/attention_pdf.pdf" 
    output_dir = "./extracted_images"
    processor = ImageTextProcessor()  
    image_paths = processor.extract_images_from_document(file_path, output_dir=output_dir)
    results = processor.extract_text_from_images(image_paths)

    for r in results:
        print(f"\nFrom: {r['image_path']}")
        print(f"Text:\n {r['content'][:200]}")
        print(f"Embedding Length is {len(r['embedding'])} and vector: {r['embedding'][:5]}")
