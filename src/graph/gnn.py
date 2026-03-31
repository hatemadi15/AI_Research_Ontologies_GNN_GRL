import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.optim as optim
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from collections import defaultdict

# Absolute paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

GRAPHPATH = os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
EMBED_OUT = os.path.join(PROCESSED_DIR, 'gnn_embeds.pt')

# Load graph data
if not os.path.exists(GRAPHPATH) or not os.path.exists(NODEMAP_PATH):
    raise FileNotFoundError(f"Missing graph files at {GRAPHPATH} or {NODEMAP_PATH}")

data = torch.load(GRAPHPATH, weights_only=False)
nodemap = torch.load(NODEMAP_PATH, weights_only=False)
inv_nodemap = {idx: term for term, idx in nodemap.items()}
num_nodes = data.num_nodes
print(f"Loaded graph: {num_nodes} nodes, {data.num_edges} edges")

# Initial node features: MiniLM embeds (384-dim)
print("Generating initial node embeddings...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Ensure we process terms in index order (0 to N-1)
terms = [inv_nodemap[i] for i in range(num_nodes)]
init_embeds = embedder.encode(terms, show_progress_bar=True)
data.x = torch.tensor(init_embeds, dtype=torch.float)

# GNN Model: 2-layer GraphSAGE
class TermGNN(torch.nn.Module):
    def __init__(self, in_dim=384, hid_dim=128, out_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Train (self-supervised: reconstruct neighbors via link pred loss)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}...")
model = TermGNN().to(device)
data = data.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(101):  # Converges fast on small graph
    optimizer.zero_grad()
    embeds = model(data.x, data.edge_index)
    # Simple link pred: positive edges vs. random negatives
    pos_edge = data.edge_index
    neg_edge = torch.randint(0, num_nodes, pos_edge.shape, device=device)
    
    # Dot product similarity
    pos_score = (embeds[pos_edge[0]] * embeds[pos_edge[1]]).sum(dim=-1)
    neg_score = (embeds[neg_edge[0]] * embeds[neg_edge[1]]).sum(dim=-1)
    
    loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + \
           F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Infer final embeds
model.eval()
with torch.no_grad():
    gnn_embeds = model(data.x, data.edge_index).cpu().numpy()

# Save: node_idx -> 64-dim embed (for relations.py swap-in)
# We save the tensor directly and the nodemap
torch.save({'embeds': gnn_embeds, 'nodemap': nodemap}, EMBED_OUT)
print(f"Saved GNN embeds: {gnn_embeds.shape} to {EMBED_OUT}")
