"""
Module 3D → 4 Bridge: GNN Drug Interaction Predictor
Graph Neural Network that learns from the KG structure to predict unknown DDIs.
Uses PyTorch Geometric (CPU-compatible).

Install: pip install torch torch_geometric --break-system-packages
Colab:   !pip install torch torch_geometric
"""

import json
import numpy as np
from knowledge_graph import DrugKnowledgeGraph
from drugbank_parser import load_data

# Try importing PyG; fall back to numpy-only mode for CPU-only environments
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[GNN] PyTorch/PyG not found — using fallback similarity model.")
    print("[GNN] Install with: pip install torch torch_geometric")


# ─────────────────────────────────────────────────
#  PyG GNN (full version)
# ─────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class DDI_GNN(nn.Module):
        """
        2-layer GraphSAGE encoder + dot-product link prediction head.
        Input:  node feature matrix X, edge_index
        Output: edge probability scores
        """
        def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 32):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden)
            self.conv2 = SAGEConv(hidden, out_channels)
            self.head  = nn.Linear(out_channels * 2, 1)

        def encode(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index)
            return x

        def decode(self, z, src, dst):
            edge_feat = torch.cat([z[src], z[dst]], dim=-1)
            return torch.sigmoid(self.head(edge_feat)).squeeze()

        def forward(self, x, edge_index, src, dst):
            z = self.encode(x, edge_index)
            return self.decode(z, src, dst)


def build_pyg_data(kg: DrugKnowledgeGraph):
    """Convert NetworkX KG → PyG Data object."""
    G = kg.G
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
    node_to_idx = {n: i for i, n in enumerate(drug_nodes)}

    # Simple one-hot features + degree
    n = len(drug_nodes)
    X = np.eye(n, dtype=np.float32)   # identity features; replace with BioBERT embeddings later

    src_list, dst_list, labels = [], [], []
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    interaction_pairs = set()
    for u, v, d in G.edges(data=True):
        if d.get("edge_type") == "interacts_with":
            if u in node_to_idx and v in node_to_idx:
                interaction_pairs.add((node_to_idx[u], node_to_idx[v]))
                interaction_pairs.add((node_to_idx[v], node_to_idx[u]))

    for i, j in all_pairs:
        src_list.append(i); dst_list.append(j)
        labels.append(1 if (i, j) in interaction_pairs else 0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.float)
    X_tensor = torch.tensor(X)

    return Data(x=X_tensor, edge_index=edge_index), label_tensor, src_list, dst_list, node_to_idx, drug_nodes


def train_gnn(kg: DrugKnowledgeGraph, epochs: int = 50):
    """Train GNN on the KG (CPU-compatible, ~1 min for sample data)."""
    if not TORCH_AVAILABLE:
        print("[GNN] PyTorch not available. Falling back to similarity model.")
        return None, None, None

    print("[GNN] Building PyG data …")
    data, labels, src, dst, node_to_idx, drug_nodes = build_pyg_data(kg)

    n_features = data.x.shape[1]
    model = DDI_GNN(in_channels=n_features, hidden=64, out_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)

    print(f"[GNN] Training for {epochs} epochs …")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(data.x, data.edge_index, src_t, dst_t)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}")

    torch.save(model.state_dict(), "gnn_model.pt")
    print("[GNN] Model saved to gnn_model.pt")
    return model, node_to_idx, drug_nodes


# ─────────────────────────────────────────────────
#  Fallback: cosine-similarity "KG-only baseline"
#  (matches the ablation table in your paper)
# ─────────────────────────────────────────────────
class KGBaselinePredictor:
    """
    Ablation row 2: KG-only baseline.
    Predicts interactions via Jaccard similarity on shared targets/categories.
    No neural network required — pure graph heuristics.
    """
    def __init__(self, kg: DrugKnowledgeGraph):
        self.kg = kg
        self.G  = kg.G

    def _neighbours(self, drug_id: str) -> set:
        return set(self.G.successors(drug_id)) | set(self.G.predecessors(drug_id))

    def predict(self, drug1_name: str, drug2_name: str) -> float:
        """Returns a [0,1] interaction score."""
        id1 = self.kg.drug_index.get(drug1_name.lower())
        id2 = self.kg.drug_index.get(drug2_name.lower())
        if id1 is None or id2 is None:
            return 0.0
        n1 = self._neighbours(id1)
        n2 = self._neighbours(id2)
        if not n1 or not n2:
            return 0.0
        return len(n1 & n2) / len(n1 | n2)   # Jaccard

    def predict_prescription(self, drug_names: list) -> list:
        results = []
        for i in range(len(drug_names)):
            for j in range(i + 1, len(drug_names)):
                score = self.predict(drug_names[i], drug_names[j])
                results.append({
                    "pair": (drug_names[i], drug_names[j]),
                    "kg_score": round(score, 4),
                    "predicted_interaction": score > 0.1,
                })
        return results


# ─────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()
    kg = DrugKnowledgeGraph()
    kg.build(data)

    print("\n── KG Baseline Predictor (ablation row 2) ──────")
    baseline = KGBaselinePredictor(kg)
    test_pairs = [
        ("Warfarin", "Aspirin"),
        ("Metformin", "Lisinopril"),
        ("Digoxin", "Furosemide"),
    ]
    for d1, d2 in test_pairs:
        score = baseline.predict(d1, d2)
        print(f"  {d1} + {d2}: KG score = {score:.4f}")

    print("\n── GNN Training (ablation row 3) ───────────────")
    model, node_to_idx, drug_nodes = train_gnn(kg, epochs=30)
    if model is None:
        print("  (install pytorch + torch_geometric on Colab for full GNN)")
