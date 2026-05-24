"""
Master Pipeline — End-to-End DDI Detection
Integrates: OCR (Module 1) → NER (Module 2) → KG Lookup (Module 3) → GNN (Module 4)
"""

import json
from drugbank_parser import load_data
from knowledge_graph import DrugKnowledgeGraph
from gnn_predictor import KGBaselinePredictor
from biobert_ner import extract_drugs_simple as extract_drugs

# ── Module 1 placeholder (swap with your TrOCR code) ────────────────────────
def mock_ocr(image_path: str = None) -> str:
    """
    Placeholder for Module 1 TrOCR output.
    Replace with: from trocr_pipeline import transcribe; return transcribe(image_path)
    """
    return (
        "Patient: John Doe  Age: 64\n"
        "Rx: Warfarin 5mg once daily\n"
        "    Aspirin 100mg once daily\n"
        "    Omeprazole 20mg once daily\n"
        "    Furosemide 40mg twice daily\n"
        "Sig: Take as directed. Follow up in 4 weeks."
    )

# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(image_path: str = None, verbose: bool = True):
    print("=" * 60)
    print("  Drug Interaction Detection Pipeline")
    print("  IEEE Paper: End-to-End NLP + KG Framework")
    print("=" * 60)

    # Step 0: Load KG
    print("\n[Step 0] Loading Knowledge Graph ...")
    data = load_data()
    kg = DrugKnowledgeGraph()
    kg.build(data)
    baseline = KGBaselinePredictor(kg)

    # Step 1: OCR
    print("\n[Step 1] OCR — Transcribing prescription ...")
    ocr_text = mock_ocr(image_path)
    if verbose:
        print("  Transcription:\n")
        for line in ocr_text.strip().split("\n"):
            print(f"    {line}")

    # Step 2: NER
    print("\n[Step 2] NER — Extracting drug names ...")
    drug_names = extract_drugs(ocr_text)
    print(f"  Extracted drugs: {drug_names}")

    # Step 3: KG Lookup
    print("\n[Step 3] KG Lookup — Checking interactions ...")
    kg_alerts = kg.check_prescription(drug_names)
    if kg_alerts:
        for alert in kg_alerts:
            sev = alert["severity"].upper()
            print(f"  ⚠️  [{sev}] {alert['pair'][0]} + {alert['pair'][1]}")
            print(f"     {alert['description'][:100]}")
    else:
        print("  No known interactions found in KG.")

    # Step 4: KG Baseline scores
    print("\n[Step 4] KG Baseline Scores (ablation row 2) ...")
    baseline_results = baseline.predict_prescription(drug_names)
    for r in baseline_results:
        flag = "⚠️ " if r["predicted_interaction"] else "✓ "
        print(f"  {flag} {r['pair'][0]} + {r['pair'][1]}: score={r['kg_score']}")

    # Step 5: Summary
    print("\n[Summary] ─────────────────────────────────────")
    print(f"  Drugs detected:        {len(drug_names)}")
    print(f"  Known DDI alerts:      {len(kg_alerts)}")
    major    = [a for a in kg_alerts if a.get("severity") == "major"]
    moderate = [a for a in kg_alerts if a.get("severity") == "moderate"]
    print(f"  Major interactions:    {len(major)}")
    print(f"  Moderate interactions: {len(moderate)}")

    if major:
        print("\n  🔴 MAJOR ALERTS — Pharmacist review required:")
        for a in major:
            print(f"     • {a['pair'][0]} + {a['pair'][1]}")

    print("\n[Pipeline complete]")
    return {
        "ocr_text": ocr_text,
        "extracted_drugs": drug_names,
        "ddi_alerts": kg_alerts,
        "baseline_scores": baseline_results,
    }


if __name__ == "__main__":
    results = run_pipeline(verbose=True)
    with open("pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to pipeline_results.json")