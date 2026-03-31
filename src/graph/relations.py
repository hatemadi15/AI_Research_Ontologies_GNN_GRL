import torch
import pandas as pd
import re
import os
import json
import spacy
from sentence_transformers import util

# Absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")
GNN_EMBEDS_PATH = os.path.join(PROCESSED_DIR, 'gnn_embeds.pt')

# Load models
print("Loading NLP models...")
nlp = spacy.load('en_core_web_sm')
# embedder = SentenceTransformer('all-MiniLM-L6-v2') # Not needed anymore

# Load GNN Embeddings
if not os.path.exists(GNN_EMBEDS_PATH):
    raise FileNotFoundError(f"Missing GNN embeds at {GNN_EMBEDS_PATH}. Run gnn.py first.")

print(f"Loading GNN embeddings from {GNN_EMBEDS_PATH}...")
gnn_data = torch.load(GNN_EMBEDS_PATH, weights_only=False)
gnn_embeds = gnn_data['embeds']  # [N_nodes, 64] numpy array
nodemap = gnn_data['nodemap']    # word -> idx
print(f"Loaded {len(nodemap)} node embeddings (dim={gnn_embeds.shape[1]})")

# PATTERNS
REL_PATTERNS = {
    'hasProperty': r'(\w+(?:\s+\w+)?)\s+(?:has|exhibits|shows|characterised by)\s+(\w+(?:\s+\w+)?)',
    'causeOf': r'(\w+(?:\s+\w+)?)\s+(?:causes|leads?\s+to?|induces|results?\s+in)\s+(\w+(?:\s+\w+)?)',
    'partOf': r'(\w+(?:\s+\w+)?)\s+(?:part|component|consists?\s+of)\s+(\w+(?:\s+\w+)?)'
}

def load_sentences():
    raw_dir = os.path.join(RAW_DIR, 'fine_grained_ner')
    sents = []
    # print(f"DEBUG: Looking for sentences in {raw_dir}")
    
    if not os.path.exists(raw_dir):
        print(f"CRITICAL ERROR: Directory {raw_dir} does not exist.")
        return []

    files = [f for f in os.listdir(raw_dir) if f.endswith('.conll')]
    for f in files:
        with open(os.path.join(raw_dir, f), encoding='utf-8') as fp:
            sent = []
            for line in fp:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if parts:
                        sent.append(parts[0])
                elif sent:
                    sents.append(' '.join(sent))
                    sent = []
            if sent: sents.append(' '.join(sent))
    
    print(f"Loaded {len(sents)} sentences.")
    return sents[:2000] 

def extract_rels():
    sents = load_sentences()
    
    # Invert clusters logic: nodemap keys are already terms
    # We use the nodemap directly for checking valid nodes
    valid_terms = set(nodemap.keys())
    
    print("Starting pattern matching with GNN similarity...")
    rels = []
    
    # BLACKLIST for cleanup
    BLACKLIST = {"the generation", "a result", "this process", "the material", "examples", "properties", "generation"}

    for doc in nlp.pipe(sents, batch_size=50):
        text_lower = doc.text.lower()
        
        for rel_type, pattern in REL_PATTERNS.items():
            for m in re.finditer(pattern, text_lower):
                try:
                    subj, obj = m.groups()
                    
                    # CLEANUP: Strip "the " or "a " from the start
                    subj = re.sub(r'^(the|a|an)\s+', '', subj, flags=re.IGNORECASE).strip()
                    obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE).strip()
                    
                    # CHECK BLACKLIST (normalized)
                    if subj.lower() in BLACKLIST or obj.lower() in BLACKLIST:
                        continue
                    
                    if subj in valid_terms and obj in valid_terms:
                        # GNN Similarity
                        idx1 = nodemap[subj]
                        idx2 = nodemap[obj]
                        vec1 = gnn_embeds[idx1]
                        vec2 = gnn_embeds[idx2]
                        
                        # Cosine sim on numpy
                        sim = util.cos_sim(vec1, vec2)[0][0].item()
                        
                        rel_entry = {'subj': subj, 'rel': rel_type, 'obj': obj, 'sim': sim}
                        rels.append(rel_entry)
                        print(f"  -> FOUND: '{subj}' --[{rel_type}]--> '{obj}' (GNN Sim: {sim:.2f})")
                except Exception:
                    continue

    if rels:
        out_path = os.path.join(PROCESSED_DIR, 'relations.csv')
        df = pd.DataFrame(rels).drop_duplicates()
        df.to_csv(out_path, index=False)
        print(f"\nSuccess! Saved {len(df)} extracted relations to {out_path}")
        
        # Show stats
        print(df.sort_values('sim', ascending=False).head(10).to_string())
    else:
        print("No relations found.")

if __name__ == '__main__':
    extract_rels()
