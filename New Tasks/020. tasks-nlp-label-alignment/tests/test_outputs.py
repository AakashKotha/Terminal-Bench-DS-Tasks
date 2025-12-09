import torch
import pytest
import os
from transformers import BertTokenizerFast

OUTPUT_PATH = "/app/output/processed_data.pt"

@pytest.fixture
def tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Output file not found."

def test_cls_sep_ignored():
    """
    Rule 1: Special tokens must be -100.
    """
    data = torch.load(OUTPUT_PATH)
    for sample in data:
        labels = sample["labels"]
        # BERT: First is CLS, Last is SEP
        assert labels[0] == -100, "First token [CLS] label must be -100."
        assert labels[-1] == -100, "Last token [SEP] label must be -100."

def test_subword_masking(tokenizer):
    """
    Rule 2 & 3: First subword gets label, others get -100.
    Case: 'Bioinformatics' -> [Bio, ##in, ##formatics]
    Labels: [FIELD_ID, -100, -100]
    """
    data = torch.load(OUTPUT_PATH)
    # We look for the sample with "Bioinformatics" (Sample ID 2)
    # We identify it by length or content
    target_sample = None
    for s in data:
        # Decode to check content
        text = tokenizer.decode(s["input_ids"])
        if "bioinformatics" in text:
            target_sample = s
            break
            
    assert target_sample is not None, "Could not find 'Bioinformatics' sample in output."
    
    input_ids = target_sample["input_ids"]
    labels = target_sample["labels"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Locate "Bio" token
    try:
        bio_idx = tokens.index("bio")
    except ValueError:
        pytest.fail("Token 'bio' not found. Tokenization might have changed?")
        
    # Expected: "bio" -> Label 2 (FIELD)
    # Next tokens "##in", "##for", "##ma", ... -> -100
    
    print(f"Tokens around 'bio': {tokens[bio_idx:bio_idx+3]}")
    print(f"Labels around 'bio': {labels[bio_idx:bio_idx+3]}")
    
    assert labels[bio_idx] == 2, f"First subword 'bio' should have label 2. Got {labels[bio_idx]}."
    assert labels[bio_idx+1] == -100, f"Subsequent subword '{tokens[bio_idx+1]}' should be -100. Got {labels[bio_idx+1]}."

def test_offset_fix():
    """
    Case 1: 'Google' is at start.
    [CLS] Google ...
    Idx 0: -100
    Idx 1: 1 (ORG)
    """
    data = torch.load(OUTPUT_PATH)
    # Find Google sample
    target_sample = None
    for s in data:
        # simplistic check
        if len(s["input_ids"]) < 10: # Google is a tech giant (short)
            target_sample = s
            break
            
    labels = target_sample["labels"]
    
    # Index 1 should be Google (ORG=1)
    assert labels[1] == 1, f"Token at index 1 ('Google') should be label 1. Got {labels[1]}. Did you handle the CLS offset?"