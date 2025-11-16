"""Mindmap (knowledge graph) generation and persistence service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from backend.core.config import ROOT_DIR, settings
from backend.services.vector_db import VectorDBService


class MindmapService:
    """Generates and serves a concise knowledge graph for the papers."""

    def __init__(self, vector_db: VectorDBService, storage_dir: Optional[Path] = None):
        self.vector_db = vector_db
        self.storage_dir = storage_dir or (ROOT_DIR / "data" / "mindmap")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.graph_file = self.storage_dir / "graph.json"

        # Configure LLM (use stable, low-temp model for structured JSON)
        genai.configure(api_key=settings.google_api_key)
        orchestrator_model_name = (
            getattr(settings.agent, "orchestrator_model", None) or settings.agent.model
        )
        self.model = genai.GenerativeModel(
            model_name=orchestrator_model_name,
            generation_config={
                "temperature": getattr(settings.agent, "orchestrator_temperature", 0.2),
                "max_output_tokens": getattr(settings.agent, "orchestrator_max_output_tokens", 2048),
            },
        )

    def build_prompt(self, summaries: List[Dict[str, Any]]) -> str:
        """Build prompt to extract a hierarchical knowledge tree as JSON (NotebookLM-style)."""
        items = []
        for s in summaries:
            # Always use the canonical title string ("<filename> (<inferred_title>)")
            title = s.get("paper_title") or s.get("paper_filename", "")
            raw_summary = s.get("summary", "") or ""
            cleaned_summary = raw_summary[:400].replace("\n", " ")
            items.append(f"- {title}: {cleaned_summary}")
        papers_block = "\n".join(items)

        # Get configuration values
        max_depth = settings.mindmap.max_depth
        min_themes = settings.mindmap.min_themes
        max_themes = settings.mindmap.max_themes
        max_node_length = settings.mindmap.node_name_max_length

        # Adjust depth guidance based on max_depth
        if max_depth <= 3:
            depth_guidance = "Keep the structure shallow with 2-3 levels total."
        elif max_depth == 4:
            depth_guidance = "Create a moderate hierarchy with 3-4 levels: organize papers under subthemes within broader themes."
        else:  # max_depth >= 5
            depth_guidance = "Create a detailed, deep hierarchy with 4-5 levels: use multiple nested subtopic layers (Theme > Major Subtopic > Minor Subtopic > Specific Area > Paper) to organize papers granularly."

        return (
            "You are an information architect. From the papers and summaries below, "
            "produce a hierarchical knowledge tree (NotebookLM-style) organized by nested topics and subtopics.\n\n"
            "Focus on *conceptual content* rather than document structure or publication type. "
            "Internal node names should capture the key ideas, phenomena, approaches, and contexts described in the papers, "
            "not generic labels.\n\n"
            f"PAPERS (only these are allowed):\n{papers_block}\n\n"
            "Output STRICTLY valid JSON (no markdown, no backticks, no explanation). Use exactly this recursive structure:\n"
            "{\n"
            '  "name": "Research Topics",\n'
            '  "children": [\n'
            '    {\n'
            '      "name": "Theme or Category",\n'
            '      "children": [\n'
            '        {\n'
            '          "name": "Subtheme",\n'
            '          "children": [\n'
            '            {"name": "Paper canonical title from the allowed list"}\n'
            '          ]\n'
            '        }\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            "}\n\n"
            "Constraints (must follow all):\n"
            "- Use ONLY the provided paper canonical titles from the list; NEVER invent or alter titles. Canonical titles are of the exact form '<filename> (<inferred_title_if_present>)'.\n"
            "- Papers must appear ONLY as leaf nodes (no children under a paper node).\n"
            f"- Create {min_themes}–{max_themes} top-level themes.\n"
            f"- Depth requirement: {depth_guidance}\n"
            f"- Maximum allowed depth is {max_depth} levels from root to paper.\n"
            "- A paper may appear under multiple branches if truly relevant, but NEVER duplicate a paper under the same branch.\n"
            f"- Keep node names concise ( {max_node_length} characters), descriptive, and ASCII-safe.\n"
            "- Do NOT use generic structural or publication-type labels as node names (for example: 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion', 'Overview', 'Review', 'State of the Art', 'Literature Review').\n"
            "- Internal node names should summarize what is being studied or addressed and, where applicable, how or in what context (e.g., combining phenomenon, signals/inputs, approaches, or settings), without mentioning that it is a section or a review.\n"
            "- Sort children alphabetically by 'name' at every level for determinism.\n"
            "- Output ONLY JSON. No markdown, no commentary, no code fences.\n"
        )

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse the first JSON object from text."""
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output")
        payload = text[start : end + 1]
        data = json.loads(payload)
        # Basic shape validation for hierarchical tree
        if "name" not in data:
            data["name"] = "Research Topics"
        data.setdefault("children", [])
        return data

    def _summaries(self) -> List[Dict[str, Any]]:
        try:
            return self.vector_db.get_paper_summaries()
        except Exception:
            return []

    def generate_graph(self) -> Dict[str, Any]:
        """Generate a hierarchical knowledge tree from current papers using the LLM."""
        summaries = self._summaries()
        if not summaries:
            return {"name": "Research Topics", "children": []}

        prompt = self.build_prompt(summaries)
        response = self.model.generate_content(prompt)
        tree = self._safe_parse_json(response.text or "")

        # Post-process the tree to normalize and de-duplicate internal concepts
        tree = self._normalize_and_deduplicate(tree)

        return tree
    
    def generate_paper_tree(self, paper_id: str) -> Dict[str, Any]:
        """Extract a subtree for a single paper from the global mindmap.
        
        This searches the global tree for all occurrences of the paper
        and returns a tree with the paper as root and all its topic contexts.
        """
        # Load the global tree
        global_tree = self.load_graph()
        
        # Get paper canonical title (or filename) from paper_id
        summaries = self._summaries()
        paper_name = None
        for s in summaries:
            if s.get("paper_id") == paper_id:
                paper_name = s.get("paper_title") or s.get("paper_filename")
                break

        if not paper_name:
            return {"name": "Paper Not Found", "children": []}

        # Search for all occurrences of this paper in the tree and collect parent topics
        topics = []
        self._find_paper_topics(global_tree, paper_name, [], topics)

        if not topics:
            return {"name": paper_name, "children": []}

        # Build a tree with paper as root and collected topics as children
        return {"name": paper_name, "children": topics}
    
    def _find_paper_topics(self, node: Dict[str, Any], paper_name: str, path: List[str], result: List[Dict[str, Any]]) -> None:
        """Recursively find all paths leading to a specific paper."""
        # If this node is the paper we're looking for
        if node.get('name') == paper_name and not node.get('children'):
            # Build topic structure from the path, skipping the root node (first element)
            if len(path) > 1:
                # Skip the first element (root node) and build from remaining path
                filtered_path = path[1:]
                
                # Convert path to nested structure
                topic = {"name": filtered_path[-1], "children": []}
                for parent_name in reversed(filtered_path[:-1]):
                    topic = {"name": parent_name, "children": [topic]}
                result.append(topic)
            return
        
        # Recursively search children
        for child in node.get('children', []):
            self._find_paper_topics(child, paper_name, path + [node['name']], result)

    def save_graph(self, tree: Dict[str, Any]) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.graph_file, "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        return self.graph_file

    def load_graph(self) -> Dict[str, Any]:
        if not self.graph_file.exists():
            return {"name": "Research Topics", "children": []}
        with open(self.graph_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def rebuild_and_persist(self) -> Dict[str, Any]:
        tree = self.generate_graph()
        self.save_graph(tree)
        return tree

    # ------------------------------------------------------------------
    # Normalization and de-duplication of concept nodes
    # ------------------------------------------------------------------

    def _normalize_and_deduplicate(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize concept node names and merge near-duplicates.

        - Paper leaves (nodes with no children) are left untouched.
        - Internal concept nodes are normalized (case/whitespace/punctuation).
        - Sibling concept nodes with the same normalized name are merged and
          their children concatenated.
        """

        def normalize_name(name: str) -> str:
            # Lowercase, strip whitespace, collapse internal spaces
            simplified = " ".join(name.strip().lower().split())
            # Strip some trivial punctuation at the ends
            return simplified.strip(".:-")

        def process_node(node: Dict[str, Any]) -> Dict[str, Any]:
            children = node.get("children") or []
            if not children:
                # Leaf (typically a paper) – do not touch
                return {"name": node.get("name", ""), "children": []}

            # First recursively process all children
            processed_children: List[Dict[str, Any]] = [process_node(c) for c in children]

            # Group non-leaf children by normalized name and merge duplicates
            grouped: Dict[str, Dict[str, Any]] = {}
            leaves: List[Dict[str, Any]] = []

            for child in processed_children:
                cname = child.get("name", "")
                cchildren = child.get("children") or []
                if not cchildren:
                    # Keep paper leaves as-is
                    leaves.append(child)
                    continue

                key = normalize_name(cname)
                if key in grouped:
                    # Merge: extend children list
                    grouped[key]["children"].extend(cchildren)
                else:
                    # First occurrence; store a copy
                    grouped[key] = {"name": cname, "children": cchildren}

            # Rebuild children list: merged concept nodes + untouched leaves
            merged_concepts = list(grouped.values())
            # Optionally, we could sort, but the LLM prompt already asks for sorting
            new_children = merged_concepts + leaves

            return {"name": node.get("name", ""), "children": new_children}

        return process_node(tree)
