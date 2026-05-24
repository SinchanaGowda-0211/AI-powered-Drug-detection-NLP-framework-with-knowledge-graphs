"""
Module 3C: Neo4j Exporter
Uploads the NetworkX knowledge graph to Neo4j for production use.

Setup:
  pip install neo4j
  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
  Then visit http://localhost:7474 to verify.
"""

import json
from knowledge_graph import DrugKnowledgeGraph
from drugbank_parser import load_data

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("[Neo4j] neo4j package not installed. Run: pip install neo4j")


# ─────────────────────────────────────────────────
#  Cypher queries
# ─────────────────────────────────────────────────
MERGE_DRUG = """
MERGE (d:Drug {drugbank_id: $id})
SET d.name = $name,
    d.categories = $categories,
    d.targets = $targets
RETURN d
"""

MERGE_TARGET = """
MERGE (t:Target {name: $name})
RETURN t
"""

MERGE_CATEGORY = """
MERGE (c:Category {name: $name})
RETURN c
"""

MERGE_INTERACTS = """
MATCH (a:Drug {drugbank_id: $id1}), (b:Drug {drugbank_id: $id2})
MERGE (a)-[r:INTERACTS_WITH {severity: $severity}]->(b)
SET r.description = $description
RETURN r
"""

MERGE_TARGETS_REL = """
MATCH (d:Drug {drugbank_id: $drug_id}), (t:Target {name: $target_name})
MERGE (d)-[:TARGETS]->(t)
"""

MERGE_CATEGORY_REL = """
MATCH (d:Drug {drugbank_id: $drug_id}), (c:Category {name: $cat_name})
MERGE (d)-[:BELONGS_TO]->(c)
"""

QUERY_INTERACTIONS = """
MATCH (a:Drug)-[r:INTERACTS_WITH]->(b:Drug)
WHERE a.name = $drug_name OR b.name = $drug_name
RETURN a.name AS drug1, b.name AS drug2, r.severity AS severity, r.description AS description
ORDER BY r.severity
"""

QUERY_PAIR = """
MATCH (a:Drug {name: $name1})-[r:INTERACTS_WITH]-(b:Drug {name: $name2})
RETURN r.severity AS severity, r.description AS description
LIMIT 1
"""


class Neo4jKGExporter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package required. Run: pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[Neo4j] Connected to {uri}")

    def close(self):
        self.driver.close()

    def _run(self, query, params=None):
        with self.driver.session() as session:
            return session.run(query, params or {})

    def clear_db(self):
        """Wipe all nodes/edges — use with caution."""
        self._run("MATCH (n) DETACH DELETE n")
        print("[Neo4j] Database cleared.")

    def upload_graph(self, kg: DrugKnowledgeGraph):
        """Upload entire NetworkX graph to Neo4j."""
        G = kg.G
        drug_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]
        target_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "target"]
        cat_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "category"]

        print(f"[Neo4j] Uploading {len(drug_nodes)} drug nodes …")
        for node_id, attrs in drug_nodes:
            self._run(MERGE_DRUG, {
                "id": node_id,
                "name": attrs.get("label", node_id),
                "categories": attrs.get("categories", []),
                "targets": attrs.get("targets", []),
            })

        print(f"[Neo4j] Uploading {len(target_nodes)} target nodes …")
        for node_id, attrs in target_nodes:
            name = attrs.get("label", node_id.replace("TGT:", ""))
            self._run(MERGE_TARGET, {"name": name})

        print(f"[Neo4j] Uploading {len(cat_nodes)} category nodes …")
        for node_id, attrs in cat_nodes:
            name = attrs.get("label", node_id.replace("CAT:", ""))
            self._run(MERGE_CATEGORY, {"name": name})

        # Edges
        interaction_edges = 0
        for u, v, data in G.edges(data=True):
            et = data.get("edge_type")
            if et == "interacts_with":
                self._run(MERGE_INTERACTS, {
                    "id1": u, "id2": v,
                    "severity": data.get("severity", "unknown"),
                    "description": data.get("description", ""),
                })
                interaction_edges += 1
            elif et == "targets":
                drug_name = G.nodes[u].get("label")
                target_name = G.nodes[v].get("label", v.replace("TGT:", ""))
                self._run(MERGE_TARGETS_REL, {"drug_id": u, "target_name": target_name})
            elif et == "belongs_to":
                cat_name = G.nodes[v].get("label", v.replace("CAT:", ""))
                self._run(MERGE_CATEGORY_REL, {"drug_id": u, "cat_name": cat_name})

        print(f"[Neo4j] Uploaded {interaction_edges} interaction edges.")
        print("[Neo4j] Upload complete.")

    def query_interactions(self, drug_name: str) -> list:
        result = self._run(QUERY_INTERACTIONS, {"drug_name": drug_name})
        return [dict(r) for r in result]

    def query_pair(self, name1: str, name2: str) -> dict:
        result = self._run(QUERY_PAIR, {"name1": name1, "name2": name2})
        records = list(result)
        if records:
            return dict(records[0])
        return {"severity": None, "description": "No interaction found"}


# ─────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Build NetworkX graph first
    data = load_data()
    kg = DrugKnowledgeGraph()
    kg.build(data)

    if NEO4J_AVAILABLE:
        try:
            exporter = Neo4jKGExporter()
            exporter.clear_db()
            exporter.upload_graph(kg)

            print("\n── Query via Cypher ──────────────────────────")
            results = exporter.query_interactions("Warfarin")
            for r in results[:3]:
                print(f"  {r['drug1']} ↔ {r['drug2']} [{r['severity']}]")

            exporter.close()
        except Exception as e:
            print(f"[Neo4j] Connection failed: {e}")
            print("[Neo4j] Make sure Neo4j is running (see docstring at top of file).")
    else:
        print("[Neo4j] Skipping upload — neo4j package not installed.")
        print("[Neo4j] NetworkX graph is fully functional without Neo4j for research/paper use.")
