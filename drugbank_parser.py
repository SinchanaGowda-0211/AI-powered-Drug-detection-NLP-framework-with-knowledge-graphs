"""
Module 3A: DrugBank XML Parser
Extracts drugs, interactions, targets, and enzymes from DrugBank full XML.
Download: https://go.drugbank.com/releases/latest#full (requires free account)
For testing without XML: uses built-in sample data automatically.
"""

import xml.etree.ElementTree as ET
import json
import os
from pathlib import Path

NS = {"db": "http://www.drugbank.ca"}

def parse_drugbank_xml(xml_path: str) -> dict:
    """Parse DrugBank XML and return structured data."""
    print(f"[Parser] Loading {xml_path} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    drugs = {}
    interactions = []

    for drug in root.findall("db:drug", NS):
        drug_id_el = drug.find("db:drugbank-id[@primary='true']", NS)
        if drug_id_el is None:
            drug_id_el = drug.find("db:drugbank-id", NS)
        if drug_id_el is None:
            continue
        drug_id = drug_id_el.text

        name_el = drug.find("db:name", NS)
        name = name_el.text if name_el is not None else "Unknown"

        desc_el = drug.find("db:description", NS)
        description = desc_el.text if desc_el is not None else ""

        # Drug categories
        categories = []
        for cat in drug.findall("db:categories/db:category/db:category", NS):
            if cat.text:
                categories.append(cat.text)

        # Targets
        targets = []
        for target in drug.findall("db:targets/db:target", NS):
            t_name = target.find("db:name", NS)
            if t_name is not None and t_name.text:
                targets.append(t_name.text)

        # Drug-drug interactions
        for interaction in drug.findall("db:drug-interactions/db:drug-interaction", NS):
            interacting_id = interaction.find("db:drugbank-id", NS)
            interacting_name = interaction.find("db:name", NS)
            description_el = interaction.find("db:description", NS)

            if interacting_id is not None:
                interactions.append({
                    "drug1_id": drug_id,
                    "drug1_name": name,
                    "drug2_id": interacting_id.text,
                    "drug2_name": interacting_name.text if interacting_name is not None else "Unknown",
                    "description": description_el.text if description_el is not None else "",
                })

        drugs[drug_id] = {
            "id": drug_id,
            "name": name,
            "description": description[:200] if description else "",
            "categories": categories[:5],
            "targets": targets[:5],
        }

    print(f"[Parser] Found {len(drugs)} drugs, {len(interactions)} interactions")
    return {"drugs": drugs, "interactions": interactions}


def get_sample_data() -> dict:
    """
    Built-in sample dataset for testing without DrugBank XML.
    Contains 20 common drugs with realistic interaction data.
    """
    drugs = {
        "DB00001": {"id": "DB00001", "name": "Warfarin", "categories": ["Anticoagulant"], "targets": ["Vitamin K epoxide reductase"]},
        "DB00002": {"id": "DB00002", "name": "Aspirin", "categories": ["NSAID", "Antiplatelet"], "targets": ["COX-1", "COX-2"]},
        "DB00003": {"id": "DB00003", "name": "Metformin", "categories": ["Biguanide", "Antidiabetic"], "targets": ["AMPK"]},
        "DB00004": {"id": "DB00004", "name": "Atorvastatin", "categories": ["Statin"], "targets": ["HMG-CoA reductase"]},
        "DB00005": {"id": "DB00005", "name": "Lisinopril", "categories": ["ACE inhibitor"], "targets": ["ACE"]},
        "DB00006": {"id": "DB00006", "name": "Amoxicillin", "categories": ["Penicillin antibiotic"], "targets": ["Penicillin-binding proteins"]},
        "DB00007": {"id": "DB00007", "name": "Omeprazole", "categories": ["Proton pump inhibitor"], "targets": ["H+/K+ ATPase"]},
        "DB00008": {"id": "DB00008", "name": "Metoprolol", "categories": ["Beta blocker"], "targets": ["Beta-1 adrenoceptor"]},
        "DB00009": {"id": "DB00009", "name": "Amlodipine", "categories": ["Calcium channel blocker"], "targets": ["L-type calcium channel"]},
        "DB00010": {"id": "DB00010", "name": "Simvastatin", "categories": ["Statin"], "targets": ["HMG-CoA reductase"]},
        "DB00011": {"id": "DB00011", "name": "Clopidogrel", "categories": ["Antiplatelet"], "targets": ["P2Y12 receptor"]},
        "DB00012": {"id": "DB00012", "name": "Digoxin", "categories": ["Cardiac glycoside"], "targets": ["Na+/K+ ATPase"]},
        "DB00013": {"id": "DB00013", "name": "Furosemide", "categories": ["Loop diuretic"], "targets": ["NKCC2"]},
        "DB00014": {"id": "DB00014", "name": "Ibuprofen", "categories": ["NSAID"], "targets": ["COX-1", "COX-2"]},
        "DB00015": {"id": "DB00015", "name": "Ciprofloxacin", "categories": ["Fluoroquinolone antibiotic"], "targets": ["DNA gyrase"]},
        "DB00016": {"id": "DB00016", "name": "Fluoxetine", "categories": ["SSRI", "Antidepressant"], "targets": ["Serotonin transporter"]},
        "DB00017": {"id": "DB00017", "name": "Methotrexate", "categories": ["Antimetabolite"], "targets": ["DHFR"]},
        "DB00018": {"id": "DB00018", "name": "Prednisone", "categories": ["Corticosteroid"], "targets": ["Glucocorticoid receptor"]},
        "DB00019": {"id": "DB00019", "name": "Insulin glargine", "categories": ["Insulin"], "targets": ["Insulin receptor"]},
        "DB00020": {"id": "DB00020", "name": "Levothyroxine", "categories": ["Thyroid hormone"], "targets": ["Thyroid hormone receptor"]},
    }

    interactions = [
        {"drug1_id": "DB00001", "drug1_name": "Warfarin", "drug2_id": "DB00002", "drug2_name": "Aspirin",
         "description": "Increased bleeding risk due to combined anticoagulant and antiplatelet effects. Monitor INR closely.", "severity": "major"},
        {"drug1_id": "DB00001", "drug1_name": "Warfarin", "drug2_id": "DB00014", "drug2_name": "Ibuprofen",
         "description": "NSAIDs can displace warfarin from plasma proteins and inhibit platelet function, increasing hemorrhage risk.", "severity": "major"},
        {"drug1_id": "DB00001", "drug1_name": "Warfarin", "drug2_id": "DB00015", "drug2_name": "Ciprofloxacin",
         "description": "Ciprofloxacin inhibits CYP1A2, reducing warfarin metabolism and increasing anticoagulant effect.", "severity": "major"},
        {"drug1_id": "DB00004", "drug1_name": "Atorvastatin", "drug2_id": "DB00010", "drug2_name": "Simvastatin",
         "description": "Combination of statins increases myopathy and rhabdomyolysis risk; avoid concurrent use.", "severity": "major"},
        {"drug1_id": "DB00002", "drug1_name": "Aspirin", "drug2_id": "DB00011", "drug2_name": "Clopidogrel",
         "description": "Dual antiplatelet therapy increases bleeding risk but is indicated in ACS; use with caution.", "severity": "moderate"},
        {"drug1_id": "DB00003", "drug1_name": "Metformin", "drug2_id": "DB00013", "drug2_name": "Furosemide",
         "description": "Furosemide may increase plasma metformin levels; monitor renal function.", "severity": "moderate"},
        {"drug1_id": "DB00007", "drug1_name": "Omeprazole", "drug2_id": "DB00011", "drug2_name": "Clopidogrel",
         "description": "Omeprazole inhibits CYP2C19, reducing clopidogrel activation and antiplatelet effect.", "severity": "major"},
        {"drug1_id": "DB00012", "drug1_name": "Digoxin", "drug2_id": "DB00013", "drug2_name": "Furosemide",
         "description": "Furosemide causes hypokalemia which potentiates digoxin toxicity. Monitor potassium levels.", "severity": "major"},
        {"drug1_id": "DB00005", "drug1_name": "Lisinopril", "drug2_id": "DB00013", "drug2_name": "Furosemide",
         "description": "Combined use may cause severe hypotension, especially at start of ACE inhibitor therapy.", "severity": "moderate"},
        {"drug1_id": "DB00016", "drug1_name": "Fluoxetine", "drug2_id": "DB00014", "drug2_name": "Ibuprofen",
         "description": "SSRIs combined with NSAIDs significantly increase GI bleeding risk.", "severity": "moderate"},
        {"drug1_id": "DB00006", "drug1_name": "Amoxicillin", "drug2_id": "DB00017", "drug2_name": "Methotrexate",
         "description": "Amoxicillin may reduce renal elimination of methotrexate, leading to toxicity.", "severity": "major"},
        {"drug1_id": "DB00008", "drug1_name": "Metoprolol", "drug2_id": "DB00009", "drug2_name": "Amlodipine",
         "description": "Additive heart rate lowering; monitor for bradycardia.", "severity": "minor"},
        {"drug1_id": "DB00018", "drug1_name": "Prednisone", "drug2_id": "DB00003", "drug2_name": "Metformin",
         "description": "Corticosteroids raise blood glucose, antagonizing antidiabetic effect of metformin.", "severity": "moderate"},
        {"drug1_id": "DB00020", "drug1_name": "Levothyroxine", "drug2_id": "DB00019", "drug2_name": "Insulin glargine",
         "description": "Thyroid hormones can alter insulin requirements; monitor blood glucose.", "severity": "moderate"},
        {"drug1_id": "DB00015", "drug1_name": "Ciprofloxacin", "drug2_id": "DB00003", "drug2_name": "Metformin",
         "description": "Ciprofloxacin may enhance hypoglycemic effects; monitor blood glucose.", "severity": "moderate"},
    ]

    return {"drugs": drugs, "interactions": interactions}


def load_data(xml_path: str = None) -> dict:
    """Load from XML if available, otherwise use sample data."""
    if xml_path and os.path.exists(xml_path):
        return parse_drugbank_xml(xml_path)
    print("[Parser] No DrugBank XML found — using built-in sample data (20 drugs, 15 interactions)")
    print("[Parser] To use full DrugBank: download from https://go.drugbank.com/releases/latest")
    return get_sample_data()


if __name__ == "__main__":
    data = load_data()
    print(f"\nDrugs loaded: {len(data['drugs'])}")
    print(f"Interactions loaded: {len(data['interactions'])}")
    # Save to JSON for other modules
    with open("drug_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved to drug_data.json")
