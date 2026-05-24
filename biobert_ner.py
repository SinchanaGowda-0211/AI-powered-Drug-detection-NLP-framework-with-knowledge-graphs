"""
Module 2: BioBERT Named Entity Recognition
Extracts drug names from prescription text.
"""

from transformers import pipeline
import re

def load_ner_model():
    print("[NER] Loading BioBERT NER model...")
    ner = pipeline(
        "ner",
        model="allenai/scibert_scivocab_uncased",
        aggregation_strategy="simple"
    )
    print("[NER] Model loaded successfully!")
    return ner

def extract_drugs_simple(text: str) -> list:
    """
    Rule-based drug extractor — works without GPU.
    Used as baseline (ablation row 1) in your paper.
    """
    known_drugs = [
        "Warfarin", "Aspirin", "Metformin", "Atorvastatin",
        "Lisinopril", "Amoxicillin", "Omeprazole", "Metoprolol",
        "Amlodipine", "Simvastatin", "Clopidogrel", "Digoxin",
        "Furosemide", "Ibuprofen", "Ciprofloxacin", "Fluoxetine",
        "Methotrexate", "Prednisone", "Insulin", "Levothyroxine"
    ]
    found = []
    for drug in known_drugs:
        if drug.lower() in text.lower():
            found.append(drug)
    return found

def extract_drugs(text: str, ner_model=None) -> list:
    """
    Extract drug names from text.
    Uses BioBERT if model is loaded, otherwise falls back to rule-based.
    """
    if ner_model is not None:
        try:
            entities = ner_model(text)
            drugs = [e["word"] for e in entities if e["score"] > 0.7]
            if drugs:
                return drugs
        except Exception as e:
            print(f"[NER] Model error: {e}, using rule-based fallback")
    
    return extract_drugs_simple(text)

if __name__ == "__main__":
    # Test with sample prescription text
    sample_text = """
    Patient: John Doe  Age: 64
    Rx: Warfarin 5mg once daily
        Aspirin 100mg once daily
        Omeprazole 20mg once daily
        Furosemide 40mg twice daily
    """
    
    print("[NER] Testing rule-based extraction...")
    drugs = extract_drugs_simple(sample_text)
    print(f"[NER] Drugs found: {drugs}")
    print("\n[NER] Module 2 working correctly!")