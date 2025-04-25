from processors.text_processor import TextProcessor

def test_text_embedding():
    processor = TextProcessor()
    text = "Artificial Intelligence transforms industries"
    result = processor.process_text(text)
    
    print("\n Text Processing Result:")
    print("Type:", result["type"])
    print("Content:", result["content"])
    print("Embedding (length):", len(result["embedding"]))  # Should be 1536 for OpenAI
    print("Embedding (first 5 values):", result["embedding"][:5])

if __name__ == "__main__":
    test_text_embedding()
