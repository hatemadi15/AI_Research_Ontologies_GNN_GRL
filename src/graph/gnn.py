"""
gnn.py - GNN Training with Multiple Architectures

Supports three architectures:
  - GraphSAGE (default): 2-layer SAGEConv (384->128->64)
  - GAT: 2-layer GATConv with multi-head attention
  - RGCN: 2-layer RGCNConv for multi-relation graphs

All trained via link prediction with:
  - PyG negative_sampling (not torch.randint)
  - RandomLinkSplit for train/val split (15% validation)
  - Dropout (0.3) for regularization
  - Early stopping with patience=20
  - Saves gnn_embeddings_{arch}.npy and gnn_embed_map.json
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv, GATConv, RGCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

# Absolute paths
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

GRAPHPATH = os.path.join(PROCESSED_DIR, 'graph_fine_auto.pt')
NODEMAP_PATH = os.path.join(PROCESSED_DIR, 'nodemap_fine_auto.pt')
EMBED_NPY_OUT = os.path.join(PROCESSED_DIR, 'gnn_embeddings.npy')
EMBED_MAP_OUT = os.path.join(PROCESSED_DIR, 'gnn_embed_map.json')
# Legacy path for backward compat with relations.py
EMBED_PT_OUT = os.path.join(PROCESSED_DIR, 'gnn_embeds.pt')


class TermGNN(torch.nn.Module):
    """2-layer GraphSAGE with dropout."""

    def __init__(self, in_dim=384, hid_dim=256, out_dim=384, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATEncoder(torch.nn.Module):
    """2-layer GAT with multi-head attention."""

    def __init__(self, in_dim=384, hidden_dim=256, out_dim=384, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class RGCNEncoder(torch.nn.Module):
    """2-layer RGCN for multi-relation graphs."""

    def __init__(self, in_dim=384, hidden_dim=256, out_dim=384, num_relations=5, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type=None):
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long,
                                    device=edge_index.device)
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


def link_pred_loss(embeds, pos_edge_index, neg_edge_index):
    """Binary cross-entropy link prediction loss."""
    pos_score = (embeds[pos_edge_index[0]] * embeds[pos_edge_index[1]]).sum(dim=-1)
    neg_score = (embeds[neg_edge_index[0]] * embeds[neg_edge_index[1]]).sum(dim=-1)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score)
    )
    return pos_loss + neg_loss


def _build_model(architecture, in_dim, device, out_dim=None):
    """Build a GNN model based on the specified architecture.

    Args:
        architecture: 'sage', 'gat', or 'rgcn'
        in_dim: Input feature dimension
        device: torch device
        out_dim: Output dimension. Default reads from config.GNN_HIDDEN_DIM (384).
                 Use 64 for legacy ablation.
    """
    if out_dim is None:
        try:
            from config import GNN_HIDDEN_DIM
            out_dim = GNN_HIDDEN_DIM
        except ImportError:
            out_dim = 384
    hid_dim = 256

    if architecture == 'sage':
        model = TermGNN(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, dropout=0.3)
    elif architecture == 'gat':
        model = GATEncoder(in_dim=in_dim, hidden_dim=hid_dim, out_dim=out_dim, heads=4, dropout=0.3)
    elif architecture == 'rgcn':
        model = RGCNEncoder(in_dim=in_dim, hidden_dim=hid_dim, out_dim=out_dim, num_relations=5, dropout=0.3)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Use 'sage', 'gat', or 'rgcn'.")
    return model.to(device)


def train_gnn(architecture=None):
    """Train GNN with link prediction, early stopping, and proper splits.

    Args:
        architecture: 'sage', 'gat', or 'rgcn'. If None, reads from config.
    """
    if architecture is None:
        from config import GNN_ARCHITECTURE
        architecture = GNN_ARCHITECTURE

    print(f"\n{'='*50}")
    print(f"Training {architecture.upper()} architecture")
    print(f"{'='*50}")

    # Load graph
    if not os.path.exists(GRAPHPATH) or not os.path.exists(NODEMAP_PATH):
        raise FileNotFoundError(
            f"Missing graph files at {GRAPHPATH} or {NODEMAP_PATH}. "
            "Run builder.py first."
        )

    data = torch.load(GRAPHPATH, weights_only=False)
    nodemap = torch.load(NODEMAP_PATH, weights_only=False)
    inv_nodemap = {idx: term for term, idx in nodemap.items()}
    num_nodes = data.num_nodes
    print(f"Loaded graph: {num_nodes} nodes, {data.num_edges} edges")

    # Use data.x directly (already has MiniLM embeddings from builder.py)
    if data.x is None or data.x.shape[0] != num_nodes:
        raise ValueError("Graph data missing node features (data.x). "
                         "Rebuild with builder.py.")

    in_dim = data.x.shape[1]
    print(f"Using existing node features: dim={in_dim}")

    # Train/val split using PyG RandomLinkSplit
    transform = RandomLinkSplit(
        num_val=0.15,
        num_test=0.0,
        is_undirected=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, _ = transform(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    model = _build_model(architecture, in_dim, device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    # Early stopping
    patience = 20
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(300):
        # Train
        model.train()
        optimizer.zero_grad()

        embeds = model(train_data.x, train_data.edge_index)

        # Positive edges from training split
        pos_edge = train_data.edge_label_index[
            :, train_data.edge_label == 1
        ] if hasattr(train_data, 'edge_label') else train_data.edge_label_index

        # Negative sampling using PyG
        neg_edge = negative_sampling(
            train_data.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge.shape[1],
        )

        train_loss = link_pred_loss(embeds, pos_edge, neg_edge)
        train_loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_embeds = model(val_data.x, val_data.edge_index)
            val_pos = val_data.edge_label_index[
                :, val_data.edge_label == 1
            ] if hasattr(val_data, 'edge_label') else val_data.edge_label_index
            val_neg = negative_sampling(
                val_data.edge_index,
                num_nodes=num_nodes,
                num_neg_samples=val_pos.shape[1],
            )
            val_loss = link_pred_loss(val_embeds, val_pos, val_neg)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss.item():.4f}, "
                  f"val_loss={val_loss.item():.4f}")

        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Infer final embeddings
    model.eval()
    # Use full graph for final embedding inference
    data_device = data.to(device)
    with torch.no_grad():
        gnn_embeds = model(data_device.x, data_device.edge_index).cpu().numpy()

    # Save with architecture suffix
    arch_npy = os.path.join(PROCESSED_DIR, f'gnn_embeddings_{architecture}.npy')
    np.save(arch_npy, gnn_embeds)
    print(f"Saved gnn_embeddings_{architecture}.npy: {gnn_embeds.shape}")

    # Also save as default gnn_embeddings.npy (for backward compat)
    np.save(EMBED_NPY_OUT, gnn_embeds)
    print(f"Saved gnn_embeddings.npy: {gnn_embeds.shape}")

    # Save embed map (term -> index for the embedding matrix)
    with open(EMBED_MAP_OUT, 'w') as f:
        json.dump(nodemap, f, indent=2)
    print(f"Saved gnn_embed_map.json: {len(nodemap)} terms")

    # Save legacy .pt format for backward compat
    torch.save({'embeds': gnn_embeds, 'nodemap': nodemap}, EMBED_PT_OUT)
    print(f"Saved gnn_embeds.pt (legacy)")

    return gnn_embeds, nodemap


def train_all_architectures():
    """Train all three GNN architectures and save embeddings for each."""
    results = {}
    for arch in ['sage', 'gat', 'rgcn']:
        try:
            embeds, nodemap = train_gnn(architecture=arch)
            results[arch] = embeds.shape
            print(f"\n  {arch}: {embeds.shape}")
        except Exception as e:
            print(f"\n  {arch} FAILED: {e}")
            results[arch] = None
    print("\n=== All architectures trained ===")
    for arch, shape in results.items():
        status = f"{shape}" if shape else "FAILED"
        print(f"  {arch}: {status}")
    return results


if __name__ == "__main__":
    import sys
    if '--all' in sys.argv:
        train_all_architectures()
    else:
        train_gnn()
