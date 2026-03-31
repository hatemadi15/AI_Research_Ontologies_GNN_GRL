import os
import pickle
import torch
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import json

# Get project root directory (2 levels up from this script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dataset")

TAXONOMY_PATH = os.path.join(PROCESSED_DIR, 'taxonomy_graph.pkl')
ONTOLOGY_PATH = os.path.join(RAW_DIR, 'ontologies', 'materiominer.owl')  # Or .txt with class list
EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

def load_discovered():
    """Load taxonomy nodes as discovered classes."""
    if not os.path.exists(TAXONOMY_PATH):
        print("Run taxonomy.py first.")
        return [], None
    with open(TAXONOMY_PATH, 'rb') as f:
        G = pickle.load(f)
    # Prioritize high-degree nodes (generics) + reps from top clusters
    nodes = list(G.nodes)
    print(f"Discovered: {len(nodes)} classes ({G.number_of_edges()} isa edges)")
    return nodes, G

def load_reference():
    """Load ontology classes from TTL, OWL, or fallback list."""
    ttl_path = os.path.join(RAW_DIR, 'ontologies', 'ontology.ttl')
    owl_path = ONTOLOGY_PATH
    
    if os.path.exists(ttl_path):
        # Parse TTL using rdflib
        from rdflib import Graph, RDF, RDFS, OWL
        g = Graph()
        g.parse(ttl_path, format='turtle')
        
        # Extract class labels (rdfs:label) for all owl:Class instances
        classes = []
        for cls in g.subjects(RDF.type, OWL.Class):
            # Get rdfs:label
            for label in g.objects(cls, RDFS.label):
                label_str = str(label)
                if label_str and not label_str.startswith('http'):
                    classes.append(label_str)
                    break
        print(f"Loaded from ontology.ttl: {len(classes)} classes")
    elif os.path.exists(owl_path):
        # pip install owlready2 if OWL
        import owlready2 as owl
        onto = owl.get_ontology(owl_path).load()
        classes = [c.label[0] if c.label else str(c) for c in onto.classes()]
        print(f"Loaded from OWL: {len(classes)} classes")
    else:
        # Fallback: known MaterioMiner classes
        classes = [
            'MicrostructureFeature', 'MechanicalProperty', 'FatigueCrack', 'AlloyComposition',
            'StressState', 'GrainStructure', 'Precipitate', 'DislocationDensity'
        ]
        print("Using fallback ref ontology; add real ontology.ttl or materiominer.owl")
    
    print(f"Reference: {len(classes)} classes")
    return classes

def simple_ontoea(discovered, ref_classes, G=None, use_llm=False):
    """OntoEA: Embed → cosine sim + structure (incoming isa = more general)."""
    disc_emb = EMBEDDER.encode(discovered)
    ref_emb = EMBEDDER.encode(ref_classes)
    alignments = []
    for i, dclass in enumerate(discovered):
        scores = util.cos_sim(disc_emb[i:i+1], ref_emb)[0]
        topk = torch.topk(scores, k=min(3, len(ref_classes))).indices
        for j in topk:
            sim = scores[j].item()
            alignments.append({
                'discovered': dclass,
                'reference': ref_classes[j],
                'similarity': sim,
                'in_degree': G.in_degree(dclass) if G else 0  # Structure bonus
            })
    df = pd.DataFrame(alignments).sort_values('similarity', ascending=False)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(os.path.join(PROCESSED_DIR, 'ontology_alignment.csv'), index=False)
    
    # LLM confirmation for ambiguous matches
    if use_llm:
        df = llm_confirm_alignments(df)
        df.to_csv(os.path.join(PROCESSED_DIR, 'ontology_alignment_llm.csv'), index=False)
        print(f"LLM confirmed alignments saved to ontology_alignment_llm.csv")
    
    return df.head(20)


# ============== Gemini LLM Integration ==============
LLM_MODE = True  # Toggle: True to use Gemini for alignment confirmation, False for ablation

def llm_confirm_alignment(discovered, reference):
    """Use Gemini to confirm if an alignment is correct."""
    import google.generativeai as genai
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return 0, "No API key"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Cheap/fast
    
    prompt = f"""Materials ontology alignment check.
Discovered term: '{discovered}'
Reference class: '{reference}'

Does the discovered term match or belong to the reference class? 
Answer: Yes/No + brief reason (one sentence)."""
    
    try:
        response = model.generate_content(prompt)
        verdict = response.text.strip()
        confirmed = 1 if 'Yes' in verdict else 0
        return confirmed, verdict
    except Exception as e:
        print(f"LLM error: {e}")
        return 0, str(e)


def llm_confirm_alignments(df):
    """Confirm ambiguous alignments (0.55-0.75 similarity) using LLM."""
    df = df.copy()
    df['llm_confirmed'] = 0
    df['llm_reason'] = ''
    
    # Focus on ambiguous matches
    ambiguous_mask = (df['similarity'] > 0.55) & (df['similarity'] < 0.75)
    ambiguous = df[ambiguous_mask].head(20)
    
    print(f"LLM confirming {len(ambiguous)} ambiguous alignments...")
    for idx, row in ambiguous.iterrows():
        confirmed, reason = llm_confirm_alignment(row['discovered'], row['reference'])
        df.loc[idx, 'llm_confirmed'] = confirmed
        df.loc[idx, 'llm_reason'] = reason
    
    confirmed_count = df['llm_confirmed'].sum()
    print(f"LLM confirmed {int(confirmed_count)} alignments")
    return df


if __name__ == '__main__':
    discovered, G = load_discovered()
    ref_classes = load_reference()
    alignments = simple_ontoea(discovered, ref_classes, G, use_llm=LLM_MODE)
    print("\nTop Matches:")
    print(alignments)
    print(f"\nFull: {os.path.join(PROCESSED_DIR, 'ontology_alignment.csv')} (F1-ready vs gold)")

