"""Mindmap (knowledge graph) generation and persistence service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import asyncio

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
        self.index_file = self.storage_dir / "graph_index.json"

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

        # In-memory cache of (paper set fingerprint, query) → tree for globals
        self._graph_cache: dict[str, dict[str, Any]] = {}
        # In-memory cache of (paper_id, query) → tree for single-paper views
        self._paper_graph_cache: dict[str, dict[str, Any]] = {}

    def reset_all(self) -> None:
        """Clear all in-memory caches.
        
        Called during global data reset to ensure stale mindmap data
        doesn't persist after papers are deleted.
        """
        self._graph_cache.clear()
        self._paper_graph_cache.clear()

    def build_prompt(self, summaries: List[Dict[str, Any]], custom_query: Optional[str] = None, rag_context: Optional[str] = None) -> str:
        """Build prompt to extract a hierarchical knowledge tree as JSON (NotebookLM-style).

        If custom_query is provided, it is appended as additional user
        instructions that may influence how the hierarchy is organized,
        while all structural constraints still apply.
        """
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

        context_block = ""
        if rag_context:
            context_block = (
                "\n\nDETAILED CONTEXT FROM SEARCH (Use this to extract specific concepts, equipment, methods, or findings):\n"
                f"{rag_context}\n"
            )

        base_prompt = (
            "You are an information architect. From the papers and summaries below, "
            "produce a hierarchical knowledge tree (NotebookLM-style) organized by nested topics and subtopics.\n\n"
            "Focus on *conceptual content* rather than document structure or publication type. "
            "Internal node names should capture the key ideas, phenomena, variables, tasks, datasets, methods, equipment, tools, and contexts described in the papers, "
            "not generic labels.\n\n"
            f"PAPERS (only these are allowed):\n{papers_block}\n"
            f"{context_block}\n"
            "Output STRICTLY valid JSON (no markdown, no backticks, no explanation). Use this recursive structure:\n"
            "{\n"
            '  "name": "Research Topics (or custom root name)",\n'
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
            "- The root node should be named 'Research Topics' by default, unless custom instructions specify otherwise.\n"
            "- Use ONLY the provided paper canonical titles from the list; NEVER invent or alter titles. Canonical titles are of the exact form '<filename> (<inferred_title_if_present>)'.\n"
            "- Papers must appear ONLY as leaf nodes (no children under a paper node).\n"
            f"- Create {min_themes}–{max_themes} top-level themes.\n"
            f"- Depth requirement: {depth_guidance}\n"
            f"- Maximum allowed depth is {max_depth} levels from root to paper.\n"
            "- A paper may appear under multiple branches if truly relevant, but NEVER duplicate a paper under the same branch.\n"
            f"- Keep node names concise ( {max_node_length} characters), descriptive, and ASCII-safe.\n"
            "- Do NOT use generic structural or publication-type labels as node names (for example: 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion', 'Overview', 'Review', 'State of the Art', 'Literature Review').\n"
            "- Internal node names should summarize what is being studied or addressed and, where applicable, how or in what context (for example, combining a phenomenon or task, relevant variables or inputs, data or datasets, methods, equipment or tools, or settings), without mentioning that it is a section or a review.\n"
            "- A child node must be a semantic refinement of its parent (a more specific aspect, subtype, mechanism, context, dataset, task, or application of the parent concept). Do NOT group unrelated topics under the same parent.\n"
            "- Sibling nodes under the same parent must share a clear unifying theme that is accurately captured by the parent's name.\n"
            "- Sort children alphabetically by 'name' at every level for determinism.\n"
            "- Output ONLY JSON. No markdown, no commentary, no code fences.\n"
        )

        if custom_query:
            custom_block = (
                "\n\nIMPORTANT CUSTOM INSTRUCTIONS:\n"
                "The user has provided specific instructions for structuring this mindmap. "
                "You MUST follow these instructions, even if they override the default root node name or structure described above:\n"
                f"{custom_query}\n"
            )
            return base_prompt + custom_block

        return base_prompt

    def build_single_paper_prompt(self, summary: Dict[str, Any], custom_query: Optional[str] = None, rag_context: Optional[str] = None) -> str:
        """Build prompt for a single-paper mindmap (paper as root node).

        The model is asked to organize the content of one paper into a
        hierarchical tree: Paper Name (root) > Main Topics > Subtopics >
        Sub-subtopics.
        """
        title = summary.get("paper_title") or summary.get("paper_filename", "Paper")
        raw_summary = summary.get("summary", "") or ""
        cleaned_summary = raw_summary.replace("\n", " ")

        max_depth = settings.mindmap.max_depth
        max_node_length = settings.mindmap.node_name_max_length

        context_block = ""
        if rag_context:
            context_block = (
                "\n\nDETAILED CONTEXT FROM SEARCH (Use this to extract specific concepts, equipment, methods, or findings):\n"
                f"{rag_context}\n"
            )

        base_prompt = (
            "You are an information architect. From the content description of a single research paper, "
            "produce a hierarchical mindmap that captures its main topics and subtopics.\n\n"
            f"PAPER (canonical title): {title}\n"
            f"SUMMARY:\n{cleaned_summary}\n"
            f"{context_block}\n"
            "Output STRICTLY valid JSON (no markdown, no backticks, no explanation) with this structure:\n"
            "{\n"
            '  "name": "Paper canonical title",\n'
            '  "children": [\n'
            '    {"name": "Main Topic", "children": [\n'
            '      {"name": "Subtopic", "children": [\n'
            '        {"name": "Sub-subtopic"}\n'
            '      ]}\n'
            '    ]}\n'
            '  ]\n'
            "}\n\n"
            "Constraints (must follow all):\n"
            f"- Use the canonical paper title EXACTLY as the root 'name': {title}.\n"
            f"- Maximum allowed depth is {max_depth} levels from root to deepest subtopic.\n"
            f"- Keep node names concise (<= {max_node_length} characters), descriptive, and ASCII-safe.\n"
            "- Do NOT use generic structural or publication-type labels as node names (for example: 'Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion', 'Overview', 'Review').\n"
            "- Internal node names should summarize what is being studied or addressed and, where applicable, how or in what context.\n"
            "- A child node must always be semantically contained within its parent (a more specific topic, phenomenon, variable, method, dataset, context, or application of the parent). Do NOT attach unrelated ideas under a topic just to reuse nodes.\n"
            "- Sibling nodes under the same parent must all be coherent subtopics of that parent concept.\n"
            "- Sort children alphabetically by 'name' at every level for determinism.\n"
            "- Output ONLY JSON. No markdown, no commentary, no code fences.\n"
        )

        if custom_query:
            custom_block = (
                "\n\nCustom user instructions for structuring the mindmap (must still obey ALL constraints above except for the root node name which must stay as the canonical paper title):\n"
                f"{custom_query}\n"
            )
            return base_prompt + custom_block

        return base_prompt

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse the first JSON object from text."""
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model output")
        
        try:
            # Use raw_decode to parse just the first valid JSON object
            # and ignore any trailing text (markdown, commentary, etc.)
            payload = text[start:]
            decoder = json.JSONDecoder()
            data, _ = decoder.raw_decode(payload)
        except json.JSONDecodeError as e:
            # Fallback: try to find the last closing brace if raw_decode fails
            # (though raw_decode is usually more robust)
            end = text.rfind("}")
            if end == -1 or end <= start:
                raise ValueError(f"JSON parsing failed: {e}")
            try:
                payload = text[start : end + 1]
                data = json.loads(payload)
            except json.JSONDecodeError as e2:
                raise ValueError(f"JSON parsing failed: {e2}")

        # Basic shape validation for hierarchical tree
        if "name" not in data:
            data["name"] = "Research Topics"
        data.setdefault("children", [])
        return data

    async def _summaries(self, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        try:
            summaries = await asyncio.to_thread(self.vector_db.get_paper_summaries)
            if paper_ids:
                allowed = set(paper_ids)
                summaries = [s for s in summaries if s.get("paper_id") in allowed]
            return summaries
        except Exception:
            return []

    async def _current_paper_fingerprint(self, paper_ids: Optional[List[str]] = None) -> str:
        """Compute a stable fingerprint for the current set of papers.

        This is used to scope cached mindmaps to a specific paper set so
        that adding/removing papers automatically invalidates older graphs.
        """
        summaries = await self._summaries(paper_ids=paper_ids)
        ids = sorted(s.get("paper_id") for s in summaries if s.get("paper_id"))
        return "|".join(ids)

    def _query_key(self, custom_query: Optional[str]) -> str:
        return (custom_query or "").strip()

    def _query_hash(self, query_key: str) -> str:
        return hashlib.sha1(query_key.encode("utf-8")).hexdigest()[:16]

    def _session_graph_path(self, session_id: str, fingerprint: str, query_key: str) -> Path:
        base = self.storage_dir / "sessions" / session_id / (fingerprint or "_none")
        base.mkdir(parents=True, exist_ok=True)
        filename = f"{self._query_hash(query_key) or 'default'}.json"
        return base / filename

    def _load_index(self) -> dict:
        if not self.index_file.exists():
            return {}
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: dict) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    async def generate_graph(self, custom_query: Optional[str] = None, paper_ids: Optional[List[str]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a hierarchical knowledge tree from current papers using the LLM.

        If custom_query is provided, the generated tree is influenced by the
        additional instructions but still obeys all structural constraints.

        The result is cached per (paper set fingerprint, custom_query) pair
        so that subsequent requests reuse existing graphs when the query and
        paper set have not changed.
        """
        summaries = await self._summaries(paper_ids=paper_ids)
        if not summaries:
            return {"name": "Research Topics", "children": []}
        # Build a fingerprint for the current paper set to scope caching.
        fingerprint = await self._current_paper_fingerprint(paper_ids=paper_ids)
        query_key = self._query_key(custom_query)

        # Reuse cache when possible for both global and session-scoped graphs.
        # Cache key is based on the current paper set fingerprint and query.
        cache_key = f"{fingerprint}:::{query_key}"

        # Always check in-memory cache first
        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        # For session-scoped graphs, attempt to reuse on-disk cache for this session
        if session_id:
            try:
                p = self._session_graph_path(session_id, fingerprint, query_key)
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        tree = json.load(f)
                    self._graph_cache[cache_key] = tree
                    return tree
            except Exception:
                pass

        # Only true global graphs (no paper_ids) can be reused from disk
        # between processes. Session-scoped graphs are fast to rebuild and
        # are not persisted to disk to avoid cross-session leakage.
        if not paper_ids:
            index = self._load_index()
            meta = index.get(cache_key)
            if meta and not query_key and self.graph_file.exists():
                try:
                    tree = await asyncio.to_thread(self.load_graph)
                    self._graph_cache[cache_key] = tree
                    return tree
                except Exception:
                    # Fall through to regeneration on parse errors
                    pass

        # Build and send prompt
        rag_context = None
        if custom_query:
            # Perform RAG search to enrich the context
            # We increase n_results to get a good spread of information
            search_results = await asyncio.to_thread(
                self.vector_db.search,
                query=custom_query, 
                n_results=getattr(settings.mindmap, "rag_context_chunks", 40),
                paper_ids=paper_ids
            )
            
            # Group chunks by paper for cleaner context
            chunks_by_paper = {}
            for chunk in search_results:
                meta = chunk.get('metadata', {})
                pid = meta.get('paper_id')
                title = meta.get('paper_title') or meta.get('paper_filename') or "Unknown Paper"
                if pid:
                    if title not in chunks_by_paper:
                        chunks_by_paper[title] = []
                    chunks_by_paper[title].append(chunk.get('content', ''))
            
            context_parts = []
            for title, chunks in chunks_by_paper.items():
                context_parts.append(f"--- Details from {title} ---")
                context_parts.extend(chunks)
            
            if context_parts:
                rag_context = "\n".join(context_parts)

        prompt = self.build_prompt(summaries, custom_query=custom_query, rag_context=rag_context)
        response = await self.model.generate_content_async(prompt)
        tree = self._safe_parse_json(self._extract_text(response))

        # Post-process the tree to normalize and de-duplicate internal concepts
        tree = self._normalize_and_deduplicate(tree)

        # Persist as graph.json and update index only for true global graphs
        if not paper_ids:
            if not query_key:
                await asyncio.to_thread(self.save_graph, tree)

            # Update on-disk index for query → metadata mapping
            index = self._load_index()
            index[cache_key] = {
                "fingerprint": fingerprint,
                "query": query_key,
                "persisted": not bool(query_key),
            }
            self._save_index(index)

        # Persist session-scoped graphs on disk for reuse across restarts
        if session_id:
            try:
                p = self._session_graph_path(session_id, fingerprint, query_key)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(tree, f, ensure_ascii=False)
            except Exception:
                pass

        # Cache in memory for faster reuse in this process (global or session)
        self._graph_cache[cache_key] = tree

        return tree
    
    async def generate_paper_tree(self, paper_id: str, custom_query: Optional[str] = None) -> Dict[str, Any]:
        """Generate a mindmap focused on a single paper.

        Instead of slicing the global tree, this builds a dedicated
        hierarchy for the selected paper using only its summary, with
        the paper's canonical title as the root node.
        """
        summaries = await self._summaries()
        paper_summary: Optional[Dict[str, Any]] = None
        for s in summaries:
            if s.get("paper_id") == paper_id:
                paper_summary = s
                break

        if not paper_summary:
            return {"name": "Paper Not Found", "children": []}

        # Check in-memory cache for this paper + query
        query_key = (custom_query or "").strip()
        cache_key = f"{paper_id}:::{query_key}"
        if cache_key in self._paper_graph_cache:
            return self._paper_graph_cache[cache_key]

        # Build and send prompt for this single paper
        rag_context = None
        if custom_query:
            # Perform RAG search to enrich the context for this single paper
            search_results = await asyncio.to_thread(
                self.vector_db.search,
                query=custom_query, 
                n_results=getattr(settings.mindmap, "rag_context_chunks", 20), # Fewer chunks for single paper
                paper_ids=[paper_id]
            )
            
            context_parts = []
            for chunk in search_results:
                context_parts.append(chunk.get('content', ''))
            
            if context_parts:
                rag_context = "\n".join(context_parts)

        prompt = self.build_single_paper_prompt(paper_summary, custom_query=custom_query, rag_context=rag_context)
        response = await self.model.generate_content_async(prompt)
        tree = self._safe_parse_json(self._extract_text(response))

        # Normalize/deduplicate concept nodes for consistency
        tree = self._normalize_and_deduplicate(tree)

        # Cache for this process so repeated views reuse the same graph
        self._paper_graph_cache[cache_key] = tree

        return tree
    
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

    async def rebuild_and_persist(self) -> Dict[str, Any]:
        tree = await self.generate_graph()
        await asyncio.to_thread(self.save_graph, tree)
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

        def process_node(node: Any) -> Dict[str, Any]:
            # Be defensive: if the model returns a bare string or other
            # primitive instead of an object, wrap it as a leaf node.
            if not isinstance(node, dict):
                return {"name": str(node), "children": []}

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

    def _extract_text(self, response) -> str:
        """Safely extract text from response."""
        try:
            return response.text or ""
        except Exception:
            # Fallback for blocked/empty responses
            try:
                candidates = getattr(response, "candidates", []) or []
                texts = []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", []) if content else []
                    for part in parts:
                        if getattr(part, "text", None):
                            texts.append(part.text)
                return "\n".join(texts)
            except Exception:
                return ""
