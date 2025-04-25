def detect_file_type(file_path):
    if file_path.endswith('.pdf'):
        return 'pdf'

    elif file_path.endswith('.docx'):
        return 'docx'

    elif file_path.endswith('.pptx'):
        return 'pptx'

    elif file_path.endswith('.xlsx'):
        return 'excel'
        
    else:
        raise ValueError("Unsupported file type")
