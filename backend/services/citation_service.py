"""Citation service for building citation graphs using Semantic Scholar API."""

import asyncio
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

from backend.services.paper_manager import PaperManager
from backend.core.config import settings


class CitationService:
    """Service for fetching citation graphs and resolving local papers using OpenAlex."""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, paper_manager: PaperManager):
        self.paper_manager = paper_manager
        self.logger = logging.getLogger(__name__)
        # Ensure cache directory exists
        self.cache_dir = settings.get_vector_db_path().parent / "cache" / "citations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def get_citation_graph(self, paper_id: str) -> Dict[str, Any]:
        """
        Build a citation graph for a specific local paper.

        Args:
            paper_id: The ID of the paper in our local system.

        Returns:
            A JSON structure compatible with the mindmap visualization:
            {
                "name": "Paper Title",
                "children": [
                    { "name": "References", "children": [...] },
                    { "name": "Cited By", "children": [...] }
                ]
            }
        """
        # 1. Get local paper details
        local_paper = self.paper_manager.get_paper(paper_id)
        if not local_paper:
            raise ValueError(f"Paper with ID {paper_id} not found")

        # Determine best search title
        search_title = local_paper.canonical_title
        
        # If canonical title follows the pattern "filename (inferred)", extract inferred
        # This is safer than rsplit which might break on titles containing parentheses
        if local_paper.filename and search_title.startswith(local_paper.filename):
            remainder = search_title[len(local_paper.filename):].strip()
            if remainder.startswith("(") and remainder.endswith(")"):
                candidate = remainder[1:-1].strip()
                if candidate:
                    search_title = candidate
        
        # Fallback: if title still looks like a filename, strip extension
        if search_title.lower().endswith(".pdf"):
            search_title = search_title[:-4]
            
        # Clean up common filename artifacts if we are forced to use filename
        # e.g. "my_paper_2024" -> "my paper 2024"
        if search_title == local_paper.filename or search_title == local_paper.filename[:-4]:
             search_title = search_title.replace("_", " ").replace("-", " ")

        # 2. Search OpenAlex
        try:
            work_id = await self._search_paper(search_title)
            if not work_id:
                return {
                    "name": local_paper.canonical_title,
                    "children": [
                        {"name": "Error: Paper not found in OpenAlex", "children": []}
                    ]
                }

            # 3. Fetch Graph Data
            graph_data = await self._fetch_details(work_id)

        except Exception as e:
            self.logger.error(f"OpenAlex API error: {e}")
            return {
                "name": local_paper.canonical_title,
                "children": [
                    {"name": f"Error fetching citations: {str(e)}", "children": []}
                ]
            }

        # 4. Build Visualization Tree
        return self._build_tree(local_paper.canonical_title, graph_data)

    async def _make_request(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> httpx.Response:
        """Make a request with retry logic."""
        retries = 3
        base_delay = 1.0
        
        for i in range(retries):
            try:
                response = await client.get(url, params=params)
                if response.status_code == 429:
                    if i == retries - 1:
                        response.raise_for_status()
                    
                    wait_time = base_delay * (2 ** i)
                    self.logger.warning(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                    print(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                     if i == retries - 1:
                        raise e
                     wait_time = base_delay * (2 ** i)
                     self.logger.warning(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                     print(f"Rate limited (429) by OpenAlex. Retrying in {wait_time}s...")
                     await asyncio.sleep(wait_time)
                     continue
                raise e
            except Exception as e:
                # Handle connection errors etc
                if i == retries - 1:
                    raise e
                wait_time = base_delay * (2 ** i)
                print(f"Request failed ({e}). Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise Exception("Max retries exceeded")

    async def _search_paper(self, title: str) -> Optional[str]:
        """Search for a paper by title and return its OpenAlex ID."""
        print(f"DEBUG: Searching OpenAlex for: {title}")
        async with httpx.AsyncClient() as client:
            params = {
                "filter": f"title.search:{title}",
                "per-page": 1
            }
            try:
                response = await self._make_request(client, f"{self.BASE_URL}/works", params=params)
                data = response.json()
                
                results = data.get("results", [])
                if results:
                    return results[0]["id"]
            except Exception as e:
                self.logger.error(f"Search failed: {e}")
            return None

    async def _fetch_details(self, work_id: str) -> Dict[str, Any]:
        """Fetch references and citations for a paper from OpenAlex."""
        short_id = work_id.split("/")[-1]
        cache_path = self.cache_dir / f"oa_{short_id}.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    print(f"DEBUG: Loading {short_id} from cache")
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache for {short_id}: {e}")

        print(f"DEBUG: Fetching details for {short_id} from OpenAlex")
        async with httpx.AsyncClient() as client:
            # 1. Fetch Work Details
            response = await self._make_request(client, f"{self.BASE_URL}/works/{work_id}", params={})
            work = response.json()
            
            # 2. Fetch References
            ref_ids = work.get("referenced_works", [])
            references = []
            
            # Batch fetch references details
            chunk_size = 50
            for i in range(0, len(ref_ids), chunk_size):
                chunk = ref_ids[i:i + chunk_size]
                # chunk contains full URLs, we need just the IDs (W...)
                chunk_short_ids = [url.split("/")[-1] for url in chunk]
                ids_str = "|".join(chunk_short_ids)
                
                try:
                    ref_resp = await self._make_request(
                        client,
                        f"{self.BASE_URL}/works", 
                        params={"filter": f"openalex_id:{ids_str}", "per-page": chunk_size}
                    )
                    ref_data = ref_resp.json().get("results", [])
                    references.extend(ref_data)
                except Exception as e:
                    print(f"DEBUG: OpenAlex ref batch failed: {e}")
                await asyncio.sleep(0.1)

            # 3. Fetch Citations (Works that reference this work)
            citations = []
            try:
                cite_resp = await self._make_request(
                    client,
                    f"{self.BASE_URL}/works",
                    params={"filter": f"referenced_works:{work_id}", "per-page": 200}
                )
                citations = cite_resp.json().get("results", [])
            except Exception as e:
                print(f"DEBUG: OpenAlex citations fetch failed: {e}")

            # 4. Normalize Data
            def normalize_work(w):
                authors = w.get("authorships", [])
                topics = w.get("topics", [])
                primary_topic = topics[0]["display_name"] if topics else "Unknown Topic"
                concepts = [c["display_name"] for c in w.get("concepts", [])[:3]]
                
                return {
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "authors": [{"name": a["author"]["display_name"]} for a in authors],
                    "primary_topic": primary_topic,
                    "concepts": concepts
                }

            data = {
                "title": work.get("title"),
                "year": work.get("publication_year"),
                "authors": [{"name": a["author"]["display_name"]} for a in work.get("authorships", [])],
                "primary_topic": work.get("topics", [])[0]["display_name"] if work.get("topics") else "Unknown Topic",
                "concepts": [c["display_name"] for c in work.get("concepts", [])[:3]],
                "references": [normalize_work(r) for r in references],
                "citations": [normalize_work(c) for c in citations]
            }
            
            # Save to cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
            except Exception as e:
                self.logger.warning(f"Failed to save cache for {work_id}: {e}")
                
            return data

    def _build_tree(self, root_title: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert API response to D3.js tree format with local resolution."""
        
        # Get all local paper titles for matching
        # We create a normalized set for fuzzy matching
        local_papers = self.paper_manager.list_papers()
        local_titles_map = {} # normalized -> paper_id
        
        for p in local_papers:
            # Normalize: lowercase, alphanumeric only
            norm = "".join(c.lower() for c in p.canonical_title if c.isalnum())
            local_titles_map[norm] = p.id
            
            # Also index the inferred title part if present
            if "(" in p.canonical_title:
                inferred = p.canonical_title.rsplit("(", 1)[1].rstrip(")")
                norm_inf = "".join(c.lower() for c in inferred if c.isalnum())
                local_titles_map[norm_inf] = p.id

        def format_node(paper_obj: Dict[str, Any]) -> Dict[str, Any]:
            title = paper_obj.get("title") or "Unknown Title"
            year = paper_obj.get("year")
            topic = paper_obj.get("primary_topic")
            concepts = paper_obj.get("concepts", [])
            primary_concept = concepts[0] if concepts else "General"
            
            authors = paper_obj.get("authors") or []
            author_str = authors[0]["name"] if authors else "Unknown"
            if len(authors) > 1:
                author_str += " et al."
            
            label = f"{title} ({author_str}, {year})" if year else f"{title} ({author_str})"
            
            # Check if local
            norm_title = "".join(c.lower() for c in title if c.isalnum())
            local_id = local_titles_map.get(norm_title)
            
            node = {
                "name": label,
                "year": year,  # Store year for grouping
                "topic": topic, # Store topic for potential grouping
                "concept": primary_concept
            }
            if local_id:
                node["local_id"] = local_id
                node["is_local"] = True
            
            return node

        # Process References (Backward)
        # Allow missing titles to be shown as "Unknown Title" to avoid hiding data
        refs = data.get("references") or []
        ref_nodes = [format_node(r) for r in refs]
        
        # Handle publisher restrictions or missing data
        if not ref_nodes:
            if data.get("references_elided"):
                ref_nodes.append({
                    "name": "⚠️ References hidden by publisher",
                    "children": []
                })
            elif data.get("referenceCount", 0) > 0:
                ref_nodes.append({
                    "name": "⚠️ Unable to load references",
                    "children": []
                })

        # Group References by Topic
        refs_by_topic = {}
        unknown_topic_refs = []

        for node in ref_nodes:
            # Skip special nodes
            if node.get("name", "").startswith("⚠️"):
                unknown_topic_refs.append(node)
                continue

            t = node.get("topic")
            if t and t != "Unknown Topic":
                if t not in refs_by_topic:
                    refs_by_topic[t] = []
                refs_by_topic[t].append(node)
            else:
                unknown_topic_refs.append(node)

        # Sort topics by count descending
        sorted_ref_topics = sorted(refs_by_topic.keys(), key=lambda k: len(refs_by_topic[k]), reverse=True)
        grouped_ref_nodes = []
        for t in sorted_ref_topics:
            nodes_in_topic = refs_by_topic[t]
            
            # Sub-group by Concept
            by_concept = {}
            for n in nodes_in_topic:
                c = n.get("concept", "General")
                if c not in by_concept:
                    by_concept[c] = []
                by_concept[c].append(n)
            
            # Sort concepts by count
            sorted_concepts = sorted(by_concept.keys(), key=lambda k: len(by_concept[k]), reverse=True)
            
            concept_groups = []
            for c in sorted_concepts:
                concept_groups.append({
                    "name": f"{c} ({len(by_concept[c])})",
                    "children": by_concept[c]
                })

            grouped_ref_nodes.append({
                "name": f"{t} ({len(nodes_in_topic)})",
                "children": concept_groups
            })
        
        if unknown_topic_refs:
            grouped_ref_nodes.append({
                "name": f"Unknown Topic ({len(unknown_topic_refs)})",
                "children": unknown_topic_refs
            })
        
        # Process Citations (Forward)
        cites = data.get("citations") or []
        cite_nodes = [format_node(c) for c in cites]

        # Group Citations by Topic
        citations_by_topic = {}
        unknown_topic_citations = []

        for node in cite_nodes:
            t = node.get("topic")
            if t and t != "Unknown Topic":
                if t not in citations_by_topic:
                    citations_by_topic[t] = []
                citations_by_topic[t].append(node)
            else:
                unknown_topic_citations.append(node)

        # Sort topics by count descending
        sorted_cite_topics = sorted(citations_by_topic.keys(), key=lambda k: len(citations_by_topic[k]), reverse=True)
        
        grouped_citation_nodes = []
        for t in sorted_cite_topics:
            nodes_in_topic = citations_by_topic[t]
            
            # Sub-group by Concept
            by_concept = {}
            for n in nodes_in_topic:
                c = n.get("concept", "General")
                if c not in by_concept:
                    by_concept[c] = []
                by_concept[c].append(n)
            
            # Sort concepts by count
            sorted_concepts = sorted(by_concept.keys(), key=lambda k: len(by_concept[k]), reverse=True)
            
            concept_groups = []
            for c in sorted_concepts:
                concept_groups.append({
                    "name": f"{c} ({len(by_concept[c])})",
                    "children": by_concept[c]
                })

            grouped_citation_nodes.append({
                "name": f"{t} ({len(nodes_in_topic)})",
                "children": concept_groups
            })
        
        if unknown_topic_citations:
            grouped_citation_nodes.append({
                "name": f"Unknown Topic ({len(unknown_topic_citations)})",
                "children": unknown_topic_citations
            })

        return {
            "name": root_title,
            "children": [
                {
                    "name": f"References ({len(ref_nodes)})",
                    "children": grouped_ref_nodes
                },
                {
                    "name": f"Cited By ({len(cite_nodes)})",
                    "children": grouped_citation_nodes
                }
            ]
        }
