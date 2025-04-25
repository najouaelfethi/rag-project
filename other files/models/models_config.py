MODEL_CONFIGS = {
    # Vision models
    "image_classifier": {
        "model_name": "google/vit-base-patch16-224",
        "task": "image-classification"
    },
    "object_detector": {
        "model_name": "facebook/detr-resnet-50",
        "task": "object-detection"
    },
    "chart_detector": {
        "model_name": "microsoft/resnet-50",  # Fine-tuned for charts
        "task": "image-classification"
    },
    
    # Text models
    "text_classifier": {
        "model_name": "distilbert-base-uncased",
        "task": "text-classification"
    },
    # OCR and Vision-Language
    "ocr": {
        "model_name": "microsoft/trocr-base-handwritten",
        "task": "image-to-text"
    },
    "vqa": {
        "model_name": "dandelin/vilt-b32-finetuned-vqa",
        "task": "visual-question-answering"
    }
}