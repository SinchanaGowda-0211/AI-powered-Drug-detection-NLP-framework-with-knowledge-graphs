"""
Module 1: TrOCR Prescription OCR Pipeline
Reads handwritten prescription images and extracts text.
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import sys
import os

def load_model():
    print("[OCR] Loading TrOCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    print("[OCR] Model loaded successfully!")
    return processor, model

def transcribe(image_path: str, processor, model) -> str:
    print(f"[OCR] Reading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"[OCR] Extracted text: {text}")
    return text

if __name__ == "__main__":
    processor, model = load_model()
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = transcribe(image_path, processor, model)
            print(f"\nResult: {result}")
        else:
            print(f"Image not found: {image_path}")
    else:
        print("[OCR] No image provided.")
        print("Usage: python trocr_pipeline.py your_image.jpg")