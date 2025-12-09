import json
import torch
from transformers import BertTokenizerFast
import os

DATA_PATH = "/app/data/raw_data.jsonl"
OUTPUT_PATH = "/app/output/processed_data.pt"

# Map labels to IDs
LABEL_MAP = {"O": 0, "ORG": 1, "FIELD": 2}

def load_data():
    data = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def align_labels(tokenizer, text, entities):
    """
    INPUT:
      text: str
      entities: list of dicts {'start', 'end', 'label'}
    OUTPUT:
      labels: list of ints (len == len(input_ids))
    """
    tokenized_inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenized_inputs["input_ids"][0]
    
    # Initialize labels with O (0)
    labels = [0] * len(input_ids)
    
    # --- BROKEN LOGIC START ---
    
    # Attempt 1: Naive character mapping
    # This fails because 'tokens' includes [CLS] at index 0, so everything is shifted.
    # Also, it doesn't handle subwords correctly (assigns label to all pieces).
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Create a char mask for entities
    char_labels = ["O"] * len(text)
    for ent in entities:
        for i in range(ent["start"], ent["end"]):
            char_labels[i] = ent["label"]
            
    # Iterate tokens and try to find them in text?
    # This approach is extremely brittle and wrong for BERT.
    
    current_char = 0
    for i, token in enumerate(tokens):
        # Skip special tokens manually? The agent might try this.
        if token in ["[CLS]", "[SEP]"]:
            continue
            
        # Clean token for matching (remove ##)
        clean_token = token.replace("##", "")
        
        # Find token in char_labels
        # If the first char of the token maps to a label, assign it.
        # BUG: 'i' here matches 'tokens' index.
        # But char_labels is 0-indexed on TEXT.
        # Without proper mapping, this is a guess.
        
        # Let's verify if the char at current pointer has a label
        if current_char < len(char_labels):
            label_str = char_labels[current_char]
            if label_str in LABEL_MAP:
                labels[i] = LABEL_MAP[label_str] # Writes to Index 'i' which includes CLS offset
                
        current_char += len(clean_token)
        
        # Additional Bug: This logic assigns the label to EVERY subword part.
        # 'Bio' -> Label
        # '##in' -> Label (Wrong, should be -100)
    
    # --- BROKEN LOGIC END ---
    
    return input_ids, torch.tensor(labels)

def process():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    raw_data = load_data()
    
    processed_samples = []
    
    print(f"Processing {len(raw_data)} samples...")
    
    for item in raw_data:
        input_ids, labels = align_labels(tokenizer, item["text"], item["entities"])
        
        # Debug Print for the agent
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(f"\nText: {item['text']}")
        print(f"Tokens: {tokens}")
        print(f"Labels: {labels.tolist()}")
        
        processed_samples.append({"input_ids": input_ids, "labels": labels})
        
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(processed_samples, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process()