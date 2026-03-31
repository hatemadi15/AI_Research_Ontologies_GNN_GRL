import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def plot_knowledge_graph():
    path = os.path.join(PROCESSED_DIR, 'relations.csv')
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(path)
    
    # Filter for high confidence only (The GNN worked, so let's show off >0.90)
    df_clean = df[df['sim'] > 0.90]
    
    print(f"Plotting {len(df_clean)} high-confidence edges...")
    if len(df_clean) == 0:
        print("No high confidence relations found. Try lowering the threshold.")
        return

    # 2. Build Graph
    G = nx.DiGraph()
    for _, row in df_clean.iterrows():
        G.add_edge(row['subj'], row['obj'], label=row['rel'])

    # 3. Visualization Layout
    plt.figure(figsize=(12, 8))
    # k regulates distance between nodes. Seed ensures reproducibility.
    pos = nx.spring_layout(G, k=1.0, seed=42)  

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='gray', arrowsize=20)
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    # Draw Edge Labels (The verbs)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title("Automated Materials Ontology (GNN-Enhanced)", fontsize=16)
    plt.axis('off')
    
    # Save
    output_path = os.path.join(PROCESSED_DIR, 'knowledge_graph.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")
    # plt.show() # Disabled for headless/agent environment

if __name__ == "__main__":
    plot_knowledge_graph()
