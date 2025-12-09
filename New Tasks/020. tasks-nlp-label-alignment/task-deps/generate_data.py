import json
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_ner_data():
    print("Generating NER Dataset...")
    
    # We create cases that force subword splitting and special token issues
    data = [
        # Case 1: Simple start (Tests [CLS] offset)
        {
            "id": "1",
            "text": "Google is a tech giant.",
            "entities": [{"label": "ORG", "start": 0, "end": 6}]
        },
        # Case 2: Subword splitting (Tests 'Bioinformatics')
        # BERT tokenizer usually splits 'Bioinformatics' -> 'Bio', '##in', '##formatics'
        {
            "id": "2",
            "text": "Studying Bioinformatics at MIT.",
            "entities": [
                {"label": "FIELD", "start": 9, "end": 23}, # Bioinformatics
                {"label": "ORG", "start": 27, "end": 30}   # MIT
            ]
        }
    ]
    
    with open(os.path.join(OUTPUT_DIR, "raw_data.jsonl"), "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print("Data generated.")

if __name__ == "__main__":
    generate_ner_data()