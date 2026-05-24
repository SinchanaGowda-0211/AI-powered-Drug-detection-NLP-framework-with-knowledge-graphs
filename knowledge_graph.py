"""
Module 3B: Drug Knowledge Graph — NetworkX Implementation
CPU-friendly, no database required.
Builds a rich graph with drug nodes, interaction edges, target nodes, and category nodes.
"""

import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers/Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from drugbank_parser import load_data


SEVERITY_COLOR = {
    "major":    "#e74c3c",
    "moderate": "#f39c12",
    "minor":    "#27ae60",
    "unknown":  "#95a5a6",
}

NODE_COLOR = {
    "drug":     "#2980b9",
    "target":   "#8e44ad",
    "category": "#16a085",
}


class DrugKnowledgeGraph:
    def __init__(self):
        self.G = nx.MultiDiGraph()   # directed multi-graph (multiple edge types)
        self.drug_index = {}         # name → id
        self.stats = {}

    # ─────────────────────────────────────────────
    #  Build
    # ─────────────────────────────────────────────
    def build(self, data: dict):
        drugs = data["drugs"]
        interactions = data["interactions"]

        print("[KG] Adding drug nodes …")
        for drug_id, drug in drugs.items():
            self.G.add_node(
                drug_id,
                label=drug["name"],
                node_type="drug",
                categories=drug.get("categories", []),
                targets=drug.get("targets", []),
            )
            self.drug_index[drug["name"].lower()] = drug_id

            # Category nodes
            for cat in drug.get("categories", []):
                cat_id = f"CAT:{cat}"
                if not self.G.has_node(cat_id):
                    self.G.add_node(cat_id, label=cat, node_type="category")
                self.G.add_edge(drug_id, cat_id, edge_type="belongs_to")

            # Target nodes
            for tgt in drug.get("targets", []):
                tgt_id = f"TGT:{tgt}"
                if not self.G.has_node(tgt_id):
                    self.G.add_node(tgt_id, label=tgt, node_type="target")
                self.G.add_edge(drug_id, tgt_id, edge_type="targets")

        print("[KG] Adding interaction edges …")
        for ix in interactions:
            d1, d2 = ix["drug1_id"], ix["drug2_id"]
            # Ensure both nodes exist (may appear only as interaction partners)
            for did, dname in [(d1, ix["drug1_name"]), (d2, ix["drug2_name"])]:
                if not self.G.has_node(did):
                    self.G.add_node(did, label=dname, node_type="drug")
                    self.drug_index[dname.lower()] = did

            self.G.add_edge(
                d1, d2,
                edge_type="interacts_with",
                severity=ix.get("severity", "unknown"),
                description=ix.get("description", ""),
            )

        self._compute_stats()
        print(f"[KG] Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self

    def _compute_stats(self):
        drug_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "drug"]
        interaction_edges = [
            (u, v, d) for u, v, d in self.G.edges(data=True)
            if d.get("edge_type") == "interacts_with"
        ]
        severity_counts = defaultdict(int)
        for _, _, d in interaction_edges:
            severity_counts[d.get("severity", "unknown")] += 1

        self.stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "drug_nodes": len(drug_nodes),
            "interaction_edges": len(interaction_edges),
            "severity_breakdown": dict(severity_counts),
        }

    # ─────────────────────────────────────────────
    #  Query
    # ─────────────────────────────────────────────
    def get_interactions(self, drug_name: str) -> list:
        """Return all DDIs for a given drug name."""
        drug_id = self.drug_index.get(drug_name.lower())
        if drug_id is None:
            return []
        results = []
        for u, v, data in self.G.edges(drug_id, data=True):
            if data.get("edge_type") == "interacts_with":
                partner_name = self.G.nodes[v].get("label", v)
                results.append({
                    "drug": drug_name,
                    "interacts_with": partner_name,
                    "severity": data.get("severity", "unknown"),
                    "description": data.get("description", ""),
                })
        return results

    def check_pair(self, drug1: str, drug2: str) -> dict:
        """Check if two drugs interact."""
        id1 = self.drug_index.get(drug1.lower())
        id2 = self.drug_index.get(drug2.lower())
        if id1 is None or id2 is None:
            return {"found": False, "reason": "One or both drugs not in KG"}

        for u, v, data in self.G.edges(data=True):
            if data.get("edge_type") == "interacts_with":
                if (u == id1 and v == id2) or (u == id2 and v == id1):
                    return {
                        "found": True,
                        "severity": data.get("severity", "unknown"),
                        "description": data.get("description", ""),
                    }
        return {"found": False, "reason": "No interaction recorded between these drugs"}

    def check_prescription(self, drug_names: list) -> list:
        """
        Given a list of drug names (from OCR/NER), return all pairwise interactions.
        This is the key integration point with Modules 1 & 2.
        """
        alerts = []
        for i in range(len(drug_names)):
            for j in range(i + 1, len(drug_names)):
                result = self.check_pair(drug_names[i], drug_names[j])
                if result["found"]:
                    alerts.append({
                        "pair": (drug_names[i], drug_names[j]),
                        "severity": result["severity"],
                        "description": result["description"],
                    })
        return alerts

    def get_drug_info(self, drug_name: str) -> dict:
        drug_id = self.drug_index.get(drug_name.lower())
        if drug_id is None:
            return {}
        node = self.G.nodes[drug_id]
        return {
            "id": drug_id,
            "name": node.get("label"),
            "categories": node.get("categories", []),
            "targets": node.get("targets", []),
            "interaction_count": sum(
                1 for _, _, d in self.G.edges(drug_id, data=True)
                if d.get("edge_type") == "interacts_with"
            ),
        }

    # ─────────────────────────────────────────────
    #  Visualise
    # ─────────────────────────────────────────────
    def visualize_drug_subgraph(self, drug_name: str, output_path: str = "subgraph.png"):
        """Render a 1-hop subgraph centred on one drug."""
        drug_id = self.drug_index.get(drug_name.lower())
        if drug_id is None:
            print(f"[KG] Drug '{drug_name}' not found")
            return

        neighbors = list(self.G.predecessors(drug_id)) + list(self.G.successors(drug_id))
        subgraph_nodes = set([drug_id] + neighbors)
        SG = self.G.subgraph(subgraph_nodes)

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(SG, seed=42, k=2)

        node_colors = []
        node_sizes = []
        for node in SG.nodes():
            nt = SG.nodes[node].get("node_type", "drug")
            node_colors.append(NODE_COLOR.get(nt, "#bdc3c7"))
            node_sizes.append(2000 if node == drug_id else 800)

        edge_colors = []
        for u, v, d in SG.edges(data=True):
            et = d.get("edge_type", "")
            if et == "interacts_with":
                edge_colors.append(SEVERITY_COLOR.get(d.get("severity", "unknown"), "#95a5a6"))
            elif et == "targets":
                edge_colors.append("#8e44ad")
            else:
                edge_colors.append("#bdc3c7")

        labels = {n: SG.nodes[n].get("label", n)[:20] for n in SG.nodes()}

        nx.draw_networkx_nodes(SG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(SG, pos, edge_color=edge_colors, arrows=True,
                               arrowsize=20, width=2, alpha=0.7)
        nx.draw_networkx_labels(SG, pos, labels, font_size=8)

        # Legend
        legend_handles = [
            mpatches.Patch(color=NODE_COLOR["drug"], label="Drug"),
            mpatches.Patch(color=NODE_COLOR["target"], label="Target"),
            mpatches.Patch(color=NODE_COLOR["category"], label="Category"),
            mpatches.Patch(color=SEVERITY_COLOR["major"], label="Major interaction"),
            mpatches.Patch(color=SEVERITY_COLOR["moderate"], label="Moderate interaction"),
        ]
        plt.legend(handles=legend_handles, loc="upper left", fontsize=9)
        plt.title(f"Knowledge Graph — {drug_name} neighbourhood", fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[KG] Subgraph saved to {output_path}")

    def visualize_interaction_network(self, output_path: str = "interaction_network.png"):
        """Render only drug-drug interaction edges (no target/category nodes)."""
        drug_nodes = [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "drug"]
        SG = nx.DiGraph()
        SG.add_nodes_from(drug_nodes)

        for u, v, d in self.G.edges(data=True):
            if d.get("edge_type") == "interacts_with":
                SG.add_edge(u, v, **d)

        # Only keep nodes that have at least one interaction edge
        connected = [n for n in SG.nodes() if SG.degree(n) > 0]
        SG = SG.subgraph(connected)

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(SG, seed=7, k=1.8)

        edge_colors = [
            SEVERITY_COLOR.get(d.get("severity", "unknown"), "#95a5a6")
            for _, _, d in SG.edges(data=True)
        ]
        labels = {n: SG.nodes[n].get("label", n) for n in SG.nodes()}
        degree = dict(SG.degree())
        node_sizes = [300 + degree[n] * 300 for n in SG.nodes()]

        nx.draw_networkx_nodes(SG, pos, node_color=NODE_COLOR["drug"],
                               node_size=node_sizes, alpha=0.85)
        nx.draw_networkx_edges(SG, pos, edge_color=edge_colors,
                               arrows=True, arrowsize=18, width=2.5, alpha=0.75)
        nx.draw_networkx_labels(SG, pos, labels, font_size=9, font_color="white", font_weight="bold")

        legend_handles = [
            mpatches.Patch(color=SEVERITY_COLOR["major"], label="Major DDI"),
            mpatches.Patch(color=SEVERITY_COLOR["moderate"], label="Moderate DDI"),
            mpatches.Patch(color=SEVERITY_COLOR["minor"], label="Minor DDI"),
        ]
        plt.legend(handles=legend_handles, fontsize=10)
        plt.title("Drug-Drug Interaction Network", fontsize=15, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[KG] Interaction network saved to {output_path}")

    # ─────────────────────────────────────────────
    #  Persist
    # ─────────────────────────────────────────────
    def save(self, path: str = "knowledge_graph.json"):
        data = nx.node_link_data(self.G)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[KG] Saved to {path}")

    def load(self, path: str = "knowledge_graph.json"):
        with open(path) as f:
            data = json.load(f)
        self.G = nx.node_link_graph(data)
        # Rebuild drug_index
        for node, attrs in self.G.nodes(data=True):
            if attrs.get("node_type") == "drug":
                self.drug_index[attrs.get("label", "").lower()] = node
        self._compute_stats()
        print(f"[KG] Loaded from {path}")
        return self


# ─────────────────────────────────────────────────
#  Demo / smoke test
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()          # auto-uses sample data if no XML
    kg = DrugKnowledgeGraph()
    kg.build(data)

    print("\n── Stats ─────────────────────────────────────────")
    for k, v in kg.stats.items():
        print(f"  {k}: {v}")

    print("\n── Interactions for Warfarin ────────────────────")
    for ix in kg.get_interactions("Warfarin"):
        sev = ix["severity"].upper()
        print(f"  [{sev}] {ix['drug']} ↔ {ix['interacts_with']}: {ix['description'][:80]}…")

    print("\n── Prescription check: Warfarin + Aspirin + Omeprazole ──")
    alerts = kg.check_prescription(["Warfarin", "Aspirin", "Omeprazole"])
    if alerts:
        for a in alerts:
            print(f"  ⚠️  {a['pair'][0]} + {a['pair'][1]} [{a['severity'].upper()}]")
            print(f"     {a['description'][:100]}")
    else:
        print("  No interactions found.")

    print("\n── Generating visualisations …")
    kg.visualize_interaction_network("interaction_network.png")
    kg.visualize_drug_subgraph("Warfarin", "warfarin_subgraph.png")

    kg.save("knowledge_graph.json")
    print("\n[Done] Module 3 complete.")
