"""Citation service for building citation graphs using Semantic Scholar API."""

import logging
from typing import Any, Dict, Optional

from backend.services.paper_manager import PaperManager
from backend.services.openalex_client import OpenAlexClient
from backend.core.config import settings


class CitationService:
    """Service for fetching citation graphs and resolving local papers using OpenAlex."""

    def __init__(self, paper_manager: PaperManager, openalex_client: Optional[OpenAlexClient] = None):
        self.paper_manager = paper_manager
        self.openalex_client = openalex_client or OpenAlexClient()
        self.logger = logging.getLogger(__name__)

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
            # Check if we already have the OpenAlex ID in metadata
            work_id = local_paper.openalex_id
            
            if not work_id:
                work_id = await self.openalex_client.search_paper(search_title)
            
            if not work_id:
                return {
                    "name": local_paper.canonical_title,
                    "children": [
                        {"name": "Error: Paper not found in OpenAlex", "children": []}
                    ]
                }

            # 3. Fetch Graph Data
            graph_data = await self.openalex_client.fetch_details(work_id)

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

    async def get_paper_metadata(self, paper_id: str, fetch_full_details: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata for a specific paper from OpenAlex.
        
        Args:
            paper_id: The ID of the paper in our local system.
            fetch_full_details: If True, fetch full references and citations (cached).
                               If False, prefer local summary metadata if available.
            
        Returns:
            Dictionary containing metadata (title, year, authors, citation_count, topics, etc.)
            or None if not found.
        """
        # 1. Get local paper details
        local_paper = self.paper_manager.get_paper(paper_id)
        if not local_paper:
            return None

        # If we already have metadata in the local paper object, return it directly!
        # This is the "Indexing-time" optimization.
        # BUT: Only if we don't need the full graph (references/citations)
        if not fetch_full_details and local_paper.openalex_id and local_paper.publication_year:
             return {
                "title": local_paper.canonical_title, # Use canonical title or fetched title?
                "year": local_paper.publication_year,
                "authors": local_paper.authors,
                "citation_count": local_paper.citation_count,
                "primary_topic": local_paper.primary_topic,
                "concepts": [], # We might not have stored concepts in metadata yet
                "url": local_paper.url,
                "paper_id": paper_id,
                "local_title": local_paper.canonical_title,
                # We might need to fetch full details if references/citations are requested
                # but for basic metadata questions, this is enough.
                # However, the agent might ask for references.
                # So let's fetch full details if we need references/citations.
             }

        # Determine best search title
        search_title = local_paper.canonical_title
        
        if local_paper.filename and search_title.startswith(local_paper.filename):
            remainder = search_title[len(local_paper.filename):].strip()
            if remainder.startswith("(") and remainder.endswith(")"):
                candidate = remainder[1:-1].strip()
                if candidate:
                    search_title = candidate
        
        if search_title.lower().endswith(".pdf"):
            search_title = search_title[:-4]
            
        if search_title == local_paper.filename or search_title == local_paper.filename[:-4]:
             search_title = search_title.replace("_", " ").replace("-", " ")

        # 2. Search OpenAlex
        try:
            work_id = local_paper.openalex_id
            if not work_id:
                work_id = await self.openalex_client.search_paper(search_title)
            
            if not work_id:
                return {
                    "title": local_paper.canonical_title,
                    "year": local_paper.publication_year or "Unknown",
                    "authors": local_paper.authors or ["Unknown"],
                    "citation_count": local_paper.citation_count or "Unknown",
                    "primary_topic": local_paper.primary_topic or "Unknown",
                    "concepts": [],
                    "url": local_paper.url,
                    "paper_id": paper_id,
                    "local_title": local_paper.canonical_title,
                    "references": [],
                    "citations": []
                }

            # 3. Fetch Details (this uses cache)
            data = await self.openalex_client.fetch_details(work_id)
            
            # 4. Format for Agent
            authors = [a["name"] for a in data.get("authors", [])]
            
            return {
                "title": data.get("title"),
                "year": data.get("year"),
                "authors": authors,
                "citation_count": data.get("citation_count", 0) if "citation_count" in data else len(data.get("citations", [])),
                "primary_topic": data.get("primary_topic"),
                "concepts": data.get("concepts", []),
                "url": data.get("url"),
                "paper_id": paper_id,
                "local_title": local_paper.canonical_title,
                "references": data.get("references", []),
                "citations": data.get("citations", [])
            }

        except Exception as e:
            self.logger.error(f"Error fetching metadata for {paper_id}: {e}")
            return {
                "title": local_paper.canonical_title,
                "year": local_paper.publication_year or "Unknown",
                "authors": local_paper.authors or ["Unknown"],
                "citation_count": local_paper.citation_count or "Unknown",
                "primary_topic": local_paper.primary_topic or "Unknown",
                "concepts": [],
                "url": local_paper.url,
                "paper_id": paper_id,
                "local_title": local_paper.canonical_title,
                "references": [],
                "citations": []
            }

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
                "concept": primary_concept,
                "citation_count": paper_obj.get("citation_count", 0),
                "url": paper_obj.get("url")
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

        # Sort nodes within topics and sort topics by total citations
        topic_stats = {}
        for t, nodes in refs_by_topic.items():
            # Sort papers within topic by citation count
            nodes.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            topic_stats[t] = sum(n.get("citation_count", 0) for n in nodes)

        # Sort topics by total citations descending
        sorted_ref_topics = sorted(refs_by_topic.keys(), key=lambda k: topic_stats[k], reverse=True)
        
        grouped_ref_nodes = []
        for t in sorted_ref_topics:
            nodes_in_topic = refs_by_topic[t]
            grouped_ref_nodes.append({
                "name": f"{t} ({len(nodes_in_topic)})",
                "children": nodes_in_topic,
                "citation_count": topic_stats[t]
            })
        
        if unknown_topic_refs:
            unknown_topic_refs.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            total_citations = sum(n.get("citation_count", 0) for n in unknown_topic_refs)
            grouped_ref_nodes.append({
                "name": f"Unknown Topic ({len(unknown_topic_refs)})",
                "children": unknown_topic_refs,
                "citation_count": total_citations
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

        # Sort nodes within topics and sort topics by total citations
        cite_topic_stats = {}
        for t, nodes in citations_by_topic.items():
            # Sort papers within topic by citation count
            nodes.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            cite_topic_stats[t] = sum(n.get("citation_count", 0) for n in nodes)

        # Sort topics by total citations descending
        sorted_cite_topics = sorted(citations_by_topic.keys(), key=lambda k: cite_topic_stats[k], reverse=True)
        
        grouped_citation_nodes = []
        for t in sorted_cite_topics:
            nodes_in_topic = citations_by_topic[t]
            grouped_citation_nodes.append({
                "name": f"{t} ({len(nodes_in_topic)})",
                "children": nodes_in_topic,
                "citation_count": cite_topic_stats[t]
            })
        
        if unknown_topic_citations:
            unknown_topic_citations.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
            total_citations = sum(n.get("citation_count", 0) for n in unknown_topic_citations)
            grouped_citation_nodes.append({
                "name": f"Unknown Topic ({len(unknown_topic_citations)})",
                "children": unknown_topic_citations,
                "citation_count": total_citations
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
