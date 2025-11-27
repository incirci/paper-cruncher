"""AI Agent service using Google Gemini with RAG pipeline."""

from typing import List, Optional, Iterator, Tuple, AsyncIterator
import asyncio

import logging
import google.generativeai as genai

from backend.core.config import settings
from backend.models.schemas import Message, MessageRole, TokenUsage
from backend.services.vector_db import VectorDBService
from backend.services.citation_service import CitationService


class AIAgent:
    """AI Agent for answering questions about papers using RAG."""

    def __init__(self, vector_db: VectorDBService, citation_service: Optional[CitationService] = None):
        """
        Initialize AI Agent with Gemini.

        Args:
            vector_db: Vector database service for semantic search
            citation_service: Optional citation service for metadata retrieval
        """
        self.vector_db = vector_db
        self.citation_service = citation_service

        # Configure Gemini
        genai.configure(api_key=settings.google_api_key)

        # Initialize main text model
        self.model = genai.GenerativeModel(
            model_name=settings.agent.model,
            generation_config={
                "temperature": settings.agent.temperature,
                "max_output_tokens": settings.agent.max_response_tokens,
            },
        )

        # Separate, stable orchestrator model (lower temp, smaller output)
        orchestrator_model_name = (
            getattr(settings.agent, "orchestrator_model", None)
            or settings.agent.model
        )
        orchestrator_temperature = getattr(settings.agent, "orchestrator_temperature", 0.2)
        orchestrator_max_tokens = getattr(settings.agent, "orchestrator_max_output_tokens", 300)

        self.orchestrator_model = genai.GenerativeModel(
            model_name=orchestrator_model_name,
            generation_config={
                "temperature": orchestrator_temperature,
                "max_output_tokens": orchestrator_max_tokens,
            },
        )

        # System prompt template
        self.system_prompt = self._create_system_prompt()
        self.logger = logging.getLogger(__name__)

    # --- Safe response helpers ---
    @staticmethod
    def _extract_text_from_response(response) -> str:
        """Safely extract concatenated text parts from a Gemini response.

        Avoids using the .text quick accessor which can raise when
        no valid Part is present (e.g., finish_reason = 1 / blocked).
        Returns an empty string if no text content can be found.
        """
        if not response:
            return ""

        try:
            candidates = getattr(response, "candidates", []) or []
        except Exception:  # noqa: BLE001
            return ""

        texts: List[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                if getattr(part, "text", None):
                    texts.append(part.text)

        return "\n".join(t.strip() for t in texts if t and t.strip()).strip()

    def _build_orchestration_prompt(self, query: str, paper_summaries: List[dict]) -> str:
        """Build the prompt for the orchestration agent."""
        # Build a list of paper titles and filenames to help the orchestrator match user queries to files
        paper_entries = [f"- {p.get('paper_title', 'Unknown')} (Filename: {p['paper_filename']})" for p in paper_summaries]
        papers_list = "\n".join(paper_entries)
        
        return f'''You are a retrieval coordinator. Analyze this question and decide how to gather information.

Available papers:
{papers_list}

Question: {query}

Create retrieval commands in this exact format:

COMMAND 1:
ACTION: fetch_comparison
PAPERS: ALL
FOCUS: dataset availability
KEYWORDS: dataset, code, github, url
DENSITY: high

STRATEGY: consolidate_all
REASONING: Need overview of all papers

Valid actions: 
1. fetch_summary: Use for high-level overviews, "list all papers", or general summaries. Ignores FOCUS.
2. fetch_details: Use for specific questions where the answer might be in just a few papers. (e.g. "Find the paper that uses LSTM"). KEYWORDS are highly recommended for specific fact retrieval.
3. fetch_comparison: Use for synthesizing specific information across multiple papers. (e.g. "Compare accuracy across all papers", "List the limitations of papers A and B", "Trace the evolution of X", "Categorize papers by Y", "Create a table of Z"). This searches for the FOCUS topic in each target paper. Use DENSITY: high for detailed extraction. KEYWORDS are highly recommended for specific fact retrieval.
4. fetch_metadata: Use for questions about citation counts, publication years, authors, venues, or impact. (e.g. "Which paper has the most citations?", "Who are the authors of paper X?", "When was paper Y published?"). Ignores FOCUS.
5. fetch_references: Use when asked to list references or what a paper cites. (e.g. "What papers does X cite?", "List references of Y"). Ignores FOCUS.
6. fetch_cited_by: Use when asked to list papers that cite a paper. (e.g. "Who cited paper X?", "List papers citing Y"). Ignores FOCUS.

For PAPERS, use: ALL or specific Filenames (e.g. paper.pdf) separated by commas. Do NOT use titles in the PAPERS field, only the Filenames listed above.
For DENSITY, use: normal (default, 5 chunks) or high (deep dive, 20 chunks). ALWAYS use 'high' for tables, comparisons, technical extraction, methodology, datasets, or specific fact-checking requests.
For KEYWORDS, use: Comma-separated list of words that MUST appear in the text (optional). Use this to filter out irrelevant chunks when searching for specific facts (e.g. 'accuracy, f1-score' or 'github, url').
'''

    def _parse_orchestration_response(self, response_text: str, paper_summaries: List[dict]) -> dict:
        """Parse the text response from the orchestrator into structured commands."""
        commands = []
        strategy = 'consolidate_all'
        reasoning = ''
        
        lines = response_text.split('\n')
        current_command = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('COMMAND'):
                if current_command:
                    commands.append(current_command)
                current_command = {}
            elif line.startswith('ACTION:'):
                current_command['action'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('PAPERS:'):
                papers_str = line.split(':', 1)[1].strip()
                if papers_str.upper() == 'ALL':
                    current_command['papers'] = [p['paper_filename'] for p in paper_summaries]
                else:
                    current_command['papers'] = [p.strip() for p in papers_str.split(',') if p.strip()]
            elif line.startswith('FOCUS:'):
                current_command['focus'] = line.split(':', 1)[1].strip()
            elif line.startswith('KEYWORDS:'):
                current_command['keywords'] = [k.strip().lower() for k in line.split(':', 1)[1].split(',') if k.strip()]
            elif line.startswith('DENSITY:'):
                current_command['density'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('STRATEGY:'):
                strategy = line.split(':', 1)[1].strip().lower()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if current_command:
            commands.append(current_command)
            
        return {'commands': commands, 'strategy': strategy, 'reasoning': reasoning}

    def _orchestrate_retrieval(
        self,
        query: str,
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, int, int]:
        """
        Synchronous Orchestrator Agent: Analyzes query and issues specific commands for information gathering.
        """
        paper_summaries = self.vector_db.get_paper_summaries()
        self._enrich_summaries(paper_summaries)

        # First, restrict to session-allowed papers if provided
        if allowed_paper_ids:
            allowed_set = set(allowed_paper_ids)
            paper_summaries = [
                p for p in paper_summaries if p.get("paper_id") in allowed_set
            ]
        
        # Further filter to specific paper if paper_id is provided
        if paper_id:
            paper_summaries = [p for p in paper_summaries if p.get('paper_id') == paper_id]
            if not paper_summaries:
                raise RuntimeError(f"Paper with ID {paper_id} not found")
        
        orchestration_prompt = self._build_orchestration_prompt(query, paper_summaries)
        
        try:
            orchestration_response = self.orchestrator_model.generate_content(
                orchestration_prompt, 
                generation_config={"max_output_tokens": getattr(settings.agent, "orchestrator_max_output_tokens", 300)}
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Orchestration failed: {str(e)}")

        response_text = self._extract_text_from_response(orchestration_response)
        if not response_text:
            raise RuntimeError("Orchestration produced empty content after successful completion.")
        
        result = self._parse_orchestration_response(response_text, paper_summaries)
        
        # Calculate tokens
        prompt_tokens = self.count_tokens(orchestration_prompt)
        response_tokens = self.count_tokens(response_text)
        
        return result, prompt_tokens, response_tokens

    async def _orchestrate_retrieval_async(
        self,
        query: str,
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, int, int]:
        """
        Async Orchestrator Agent: Analyzes query and issues specific commands for information gathering.
        """
        paper_summaries = await asyncio.to_thread(self.vector_db.get_paper_summaries)
        self._enrich_summaries(paper_summaries)

        # First, restrict to session-allowed papers if provided
        if allowed_paper_ids:
            allowed_set = set(allowed_paper_ids)
            paper_summaries = [
                p for p in paper_summaries if p.get("paper_id") in allowed_set
            ]
        
        # Further filter to specific paper if paper_id is provided
        if paper_id:
            paper_summaries = [p for p in paper_summaries if p.get('paper_id') == paper_id]
            if not paper_summaries:
                raise RuntimeError(f"Paper with ID {paper_id} not found")
        
        orchestration_prompt = self._build_orchestration_prompt(query, paper_summaries)
        
        try:
            orchestration_response = await self.orchestrator_model.generate_content_async(
                orchestration_prompt, 
                generation_config={"max_output_tokens": getattr(settings.agent, "orchestrator_max_output_tokens", 300)}
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Orchestration failed: {str(e)}")

        response_text = self._extract_text_from_response(orchestration_response)
        if not response_text:
            raise RuntimeError("Orchestration produced empty content after successful completion.")
        
        result = self._parse_orchestration_response(response_text, paper_summaries)
        
        # Calculate tokens
        prompt_tokens = self.count_tokens(orchestration_prompt)
        response_tokens = self.count_tokens(response_text)
        
        return result, prompt_tokens, response_tokens

    async def _execute_worker_commands(
        self,
        commands: List[dict],
        allowed_paper_ids: Optional[List[str]] = None,
        user_query: str = "",
    ) -> List[dict]:
        """Execute retrieval commands against the vector DB."""
        all_chunks = []
        
        # Get map of filename -> paper_id
        papers = await asyncio.to_thread(self.vector_db.get_all_papers)
        filename_to_id = {p['paper_filename']: p['paper_id'] for p in papers}
        
        for cmd in commands:
            action = cmd.get('action')
            target_papers = cmd.get('papers', [])
            focus = cmd.get('focus', '')
            density = cmd.get('density', 'normal')
            
            # Resolve paper IDs
            target_ids = []
            for name in target_papers:
                pid = filename_to_id.get(name)
                if pid:
                    if allowed_paper_ids is None or pid in allowed_paper_ids:
                        target_ids.append(pid)
            
            # Handle summary requests differently: fetch explicit summaries for ALL requested papers
            if action == 'fetch_summary':
                # Use the specialized summary retrieval method
                # If target_ids is empty (e.g. "ALL" or fallback), use allowed_paper_ids or None (all)
                effective_ids = target_ids if target_ids else allowed_paper_ids
                
                # If effective_ids is None (ALL), resolve to actual IDs to detect missing ones
                if effective_ids is None and self.citation_service:
                    effective_ids = [p.id for p in self.citation_service.paper_manager.list_papers()]
                    if allowed_paper_ids:
                        effective_ids = [pid for pid in effective_ids if pid in allowed_paper_ids]

                summaries = await asyncio.to_thread(self.vector_db.get_paper_summaries, paper_ids=effective_ids)
                found_ids = {s['paper_id'] for s in summaries}
                
                for s in summaries:
                    pid = s['paper_id']
                    content = f"Micro-Summary of {s['paper_title']}:\n{s['summary']}"
                    
                    # Enrich with metadata if available
                    if self.citation_service:
                        try:
                            # Use fetch_full_details=False to use local cache/basic info only
                            meta = await self.citation_service.get_paper_metadata(pid, fetch_full_details=False)
                            if meta:
                                meta_str = (
                                    f"\n\nMetadata:\n"
                                    f"- Year: {meta.get('year', 'N/A')}\n"
                                    f"- Authors: {', '.join(meta.get('authors', []))}\n"
                                    f"- Citations: {meta.get('citation_count', 'N/A')}\n"
                                    f"- Topic: {meta.get('primary_topic', 'N/A')}"
                                )
                                content += meta_str
                        except Exception:
                            pass # Fail gracefully on metadata fetch

                    all_chunks.append({
                        "id": f"summary_{s['paper_id']}",
                        "content": content,
                        "metadata": {
                            "paper_id": s['paper_id'],
                            "paper_title": s['paper_title'],
                            "paper_filename": s['paper_filename'],
                            "source": "micro_summary"
                        }
                    })
                
                # Fallback for papers missing from VectorDB
                if effective_ids and self.citation_service:
                    for pid in effective_ids:
                        if pid not in found_ids:
                            try:
                                meta = await self.citation_service.get_paper_metadata(pid, fetch_full_details=False)
                                if meta:
                                    content = (
                                        f"Micro-Summary of {meta.get('title', 'Unknown')}:\n"
                                        f"(Content not indexed, metadata only)\n"
                                        f"- Year: {meta.get('year', 'N/A')}\n"
                                        f"- Authors: {', '.join(meta.get('authors', []))}\n"
                                        f"- Citations: {meta.get('citation_count', 'N/A')}\n"
                                        f"- Topic: {meta.get('primary_topic', 'N/A')}"
                                    )
                                    all_chunks.append({
                                        "id": f"summary_fallback_{pid}",
                                        "content": content,
                                        "metadata": {
                                            "paper_id": pid,
                                            "paper_title": meta.get('title'),
                                            "paper_filename": meta.get('local_title'),
                                            "source": "metadata_fallback"
                                        }
                                    })
                            except Exception:
                                pass
                continue

            # Handle metadata requests
            if action == 'fetch_metadata' and self.citation_service:
                effective_ids = target_ids if target_ids else allowed_paper_ids
                if effective_ids:
                    for pid in effective_ids:
                        meta = await self.citation_service.get_paper_metadata(pid)
                        if meta:
                            content = (
                                f"Metadata for {meta['title']}:\n"
                                f"- Year: {meta['year']}\n"
                                f"- Authors: {', '.join(meta['authors'])}\n"
                                f"- Citation Count: {meta['citation_count']}\n"
                                f"- Primary Topic: {meta['primary_topic']}\n"
                                f"- Concepts: {', '.join(meta['concepts'])}\n"
                                f"- URL: {meta['url']}\n"
                            )
                            all_chunks.append({
                                "id": f"metadata_{pid}",
                                "content": content,
                                "metadata": {
                                    "paper_id": pid,
                                    "paper_title": meta['title'],
                                    "paper_filename": meta['local_title'],
                                    "source": "openalex_metadata"
                                }
                            })
                continue

            # Handle references requests
            if action == 'fetch_references' and self.citation_service:
                effective_ids = target_ids if target_ids else allowed_paper_ids
                if effective_ids:
                    for pid in effective_ids:
                        meta = await self.citation_service.get_paper_metadata(pid, fetch_full_details=True)
                        if meta:
                            refs = meta.get('references', [])
                            # Format list of references
                            refs_list = "\n".join([f"- {r.get('title')} ({r.get('year')}, {r.get('citation_count')} citations)" for r in refs])
                            
                            if not refs_list:
                                if meta.get('citation_count') == 'Unknown':
                                    refs_list = "Reference data not available (Paper not found in external database)."
                                else:
                                    refs_list = "No references found."
                                
                            content = (
                                f"References for {meta['title']}:\n"
                                f"{refs_list}\n"
                            )
                            all_chunks.append({
                                "id": f"refs_{pid}",
                                "content": content,
                                "metadata": {
                                    "paper_id": pid,
                                    "paper_title": meta['title'],
                                    "paper_filename": meta['local_title'],
                                    "source": "openalex_references"
                                }
                            })
                continue

            # Handle cited_by requests
            if action == 'fetch_cited_by' and self.citation_service:
                effective_ids = target_ids if target_ids else allowed_paper_ids
                if effective_ids:
                    for pid in effective_ids:
                        meta = await self.citation_service.get_paper_metadata(pid, fetch_full_details=True)
                        if meta:
                            cites = meta.get('citations', [])
                            # Format list of citations
                            cites_list = "\n".join([f"- {c.get('title')} ({c.get('year')}, {c.get('citation_count')} citations)" for c in cites])
                            
                            if not cites_list:
                                if meta.get('citation_count') == 'Unknown':
                                    cites_list = "Citation data not available (Paper not found in external database)."
                                else:
                                    cites_list = "No citations found."
                                
                            content = (
                                f"Papers citing {meta['title']}:\n"
                                f"{cites_list}\n"
                            )
                            all_chunks.append({
                                "id": f"cited_by_{pid}",
                                "content": content,
                                "metadata": {
                                    "paper_id": pid,
                                    "paper_title": meta['title'],
                                    "paper_filename": meta['local_title'],
                                    "source": "openalex_cited_by"
                                }
                            })
                continue

            # Handle comparison requests: ensure distribution across papers
            # Instead of a global top-N search, we search each paper individually to ensure coverage.
            if action == 'fetch_comparison' and target_ids:
                # Determine chunks per paper. 
                # If density is high, take more. If normal, take fewer to avoid context explosion.
                chunks_per_paper = 8 if density == 'high' else 3
                keywords = cmd.get('keywords', [])
                
                if keywords:
                    self.logger.info(f"Executing comparison search with keywords: {keywords}")
                
                for pid in target_ids:
                    # If keywords are provided, fetch more candidates to allow for filtering
                    n_candidates = chunks_per_paper * 3 if keywords else chunks_per_paper
                    
                    results = await asyncio.to_thread(
                        self.vector_db.search,
                        query=f"{focus} {user_query}",
                        n_results=n_candidates,
                        paper_ids=[pid]
                    )
                    
                    if keywords:
                        # Filter results: keep chunk if it contains ANY of the keywords
                        filtered_results = []
                        for chunk in results:
                            content_lower = chunk['content'].lower()
                            if any(k in content_lower for k in keywords):
                                filtered_results.append(chunk)
                        
                        # Take top N from filtered
                        all_chunks.extend(filtered_results[:chunks_per_paper])
                    else:
                        all_chunks.extend(results)
                continue

            # Construct search query
            search_query = f"{focus} {user_query}"
            
            # Determine n_results based on density
            base_chunks = settings.chunking.max_chunks_per_query
            n_results = (base_chunks * 4) if density == 'high' else base_chunks
            
            keywords = cmd.get('keywords', [])
            if keywords:
                self.logger.info(f"Executing search with keywords: {keywords}")
                # Fetch more candidates for filtering
                search_n_results = n_results * 3
            else:
                search_n_results = n_results

            # Execute search
            results = await asyncio.to_thread(
                self.vector_db.search,
                query=search_query,
                n_results=search_n_results,
                paper_ids=target_ids if target_ids else allowed_paper_ids
            )
            
            if keywords:
                # Filter results
                filtered_results = []
                for chunk in results:
                    content_lower = chunk['content'].lower()
                    if any(k in content_lower for k in keywords):
                        filtered_results.append(chunk)
                all_chunks.extend(filtered_results[:n_results])
            else:
                all_chunks.extend(results)
            
        # Deduplicate chunks by ID
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk['id'] not in seen_ids:
                seen_ids.add(chunk['id'])
                unique_chunks.append(chunk)
                
        return unique_chunks

    async def _build_context(
        self,
        chunks: List[dict],
        include_overview: bool = True,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> str:
        """Format chunks into a context string."""
        
        # Helper to get paper details
        def get_paper_basic_details(pid, default_title, default_filename):
            if self.citation_service:
                paper = self.citation_service.paper_manager.get_paper(pid)
                if paper:
                    return paper.display_title, paper.filename
            return default_title or default_filename or "Unknown Paper", default_filename or "Unknown Filename"

        context_parts = []

        if include_overview:
            # Add list of available papers to context so the model knows what's in the library
            # regardless of what chunks were retrieved.
            papers = await asyncio.to_thread(self.vector_db.get_all_papers)
            if allowed_paper_ids:
                papers = [p for p in papers if p['paper_id'] in allowed_paper_ids]
            
            if papers:
                context_parts.append("Available Papers in Knowledge Base:")
                for p in papers:
                    title, filename = get_paper_basic_details(p['paper_id'], p.get('paper_title'), p.get('paper_filename'))
                    context_parts.append(f"- Title: {title} (Filename: {filename})")
                context_parts.append("")
        
        # Group by paper
        chunks_by_paper = {}
        for chunk in chunks:
            pid = chunk['metadata']['paper_id']
            if pid not in chunks_by_paper:
                chunks_by_paper[pid] = []
            chunks_by_paper[pid].append(chunk)
            
        for pid, paper_chunks in chunks_by_paper.items():
            # Get paper title/filename from first chunk metadata
            meta = paper_chunks[0]['metadata']
            title, filename = get_paper_basic_details(pid, meta.get('paper_title'), meta.get('paper_filename'))
            
            context_parts.append(f"--- Paper Analysis ---")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"Filename: {filename}")
            
            # Try to fetch rich metadata if available
            if self.citation_service:
                try:
                    paper_meta = await self.citation_service.get_paper_metadata(pid, fetch_full_details=False)
                    if paper_meta:
                        context_parts.append(f"Year: {paper_meta.get('year', 'N/A')}")
                        context_parts.append(f"Authors: {', '.join(paper_meta.get('authors', []))}")
                        context_parts.append(f"Citations: {paper_meta.get('citation_count', 'N/A')}")
                except Exception:
                    pass

            context_parts.append("Content Snippets:")
            for chunk in paper_chunks:
                context_parts.append(chunk['content'])
            context_parts.append("")
            
        return "\n".join(context_parts)

    def _extract_source_papers(self, chunks: List[dict]) -> List[str]:
        """Extract unique paper titles from chunks."""
        papers = set()
        for chunk in chunks:
            meta = chunk['metadata']
            title = meta.get('paper_title', meta.get('paper_filename'))
            if title:
                papers.add(title)
        return list(papers)

    def _enhance_query(self, query: str) -> str:
        """Enhance query if needed."""
        return query

    def _build_prompt(self, query: str, context: str, conversation_history: List[Message]) -> str:
        """Construct the final prompt for the model."""
        # Format history
        history_text = ""
        if conversation_history:
            history_text = "Conversation History:\n"
            for msg in conversation_history[-5:]: # Limit history
                role = "User" if msg.role == MessageRole.USER else "Assistant"
                content = msg.content
                # Truncate very long messages to save tokens
                if len(content) > 1500:
                    content = content[:1500] + "... (truncated)"
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        return f'''{self.system_prompt}

{history_text}
Context Information:
{context}

Question: {query}

Answer:'''

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model."""
        try:
            return self.model.count_tokens(text).total_tokens
        except Exception:
            return len(text) // 4

    async def count_tokens_async(self, text: str) -> int:
        """Count tokens in text using the model asynchronously."""
        try:
            resp = await self.model.count_tokens_async(text)
            return resp.total_tokens
        except Exception:
            return len(text) // 4

    # --- Imagen integration helpers ---
    def is_visualization_request(self, query: str) -> bool:
        """Heuristic to detect if the user asks for a visual/plot/diagram/image."""
        q = (query or "").lower()
        keywords = [
            "plot", "chart", "graph", "diagram", "visualize", "visualisation", "visualization",
            "draw", "sketch", "figure", "image", "picture", "illustration", "mind map", "mindmap"
        ]
        return any(k in q for k in keywords)

    async def _build_image_prompt(
        self,
        query: str,
        conversation_history: List[Message],
        paper_id: Optional[str] = None,
        max_context_chars: int = 2000,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Build an Imagen prompt, leveraging RAG context to guide the visual content."""
        # Reuse orchestrator-worker to gather key chunks, respecting session paper scope
        orchestration, _, _ = await self._orchestrate_retrieval_async(
            query,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )
        relevant_chunks = await self._execute_worker_commands(
            orchestration["commands"],
            allowed_paper_ids=allowed_paper_ids,
            user_query=query,
        )
        context = await self._build_context(
            relevant_chunks,
            include_overview=True,
            allowed_paper_ids=allowed_paper_ids,
        )
        source_papers = self._extract_source_papers(relevant_chunks)

        # Trim context to keep prompt concise
        trimmed_context = context[:max_context_chars]

        prompt = (
            "Create a clean, information-dense diagram or chart that satisfies the user's request.\n"
            "Requirements:\n"
            "- High contrast, readable labels, no watermark.\n"
            "- Balanced layout and consistent styling.\n"
            "- If hierarchical, use a tidy tree or radial layout.\n"
            "- If comparative, use a simple bar/line chart.\n"
            "- Avoid tiny text; prioritize clarity over decoration.\n\n"
            f"User request: {query}\n\n"
            "Context to ground the visualization (summarized):\n"
            f"{trimmed_context}"
        )
        return prompt, source_papers

    async def generate_image_bytes(
        self,
        prompt: str,
        mime_type: str,
        width: int,
        height: int,
        model_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate an image via Imagen and return (mime_type, base64_data).

        Uses the Imagen model via generate_content and extracts inline_data parts
        that contain image bytes (base64). No response_mime_type override.
        """
        image_model_name = model_name or getattr(settings.image, "model", "imagegeneration")
        model = genai.GenerativeModel(model_name=image_model_name)

        # Provide gentle guidance on size in the textual prompt since some SDKs
        # don't accept dimension params on generate_content
        size_hint = f"Desired size: approximately {width}x{height} pixels."
        full_prompt = f"{prompt}\n\n{size_hint}"

        try:
            response = await model.generate_content_async(full_prompt)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Imagen content generation failed: {e}")

        # Walk candidates/parts to find inline_data with image/* mime
        b64 = None
        out_mime = None
        try:
            candidates = getattr(response, "candidates", []) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content else []
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        mt = getattr(inline, "mime_type", None) or "image/png"
                        if str(mt).startswith("image/"):
                            b64 = inline.data
                            out_mime = mt
                            break
                if b64:
                    break
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not parse Imagen response: {e}")

        if not b64:
            raise RuntimeError("Imagen did not return inline image data.")

        return out_mime or (mime_type or "image/png"), b64

    def _create_system_prompt(self) -> str:
        """Create system prompt for the agent."""
        return """You are an AI research assistant specialized in analyzing journal articles and academic papers.

Your role:
- Answer questions about the content of research papers in the knowledge base
- Provide accurate, well-cited responses based on the papers
- Synthesize information across multiple papers when relevant
- Always mention which paper(s) you're referencing in your answers
- If information isn't available in the papers, clearly state that
- IMPORTANT: Pay attention to the conversation history provided below and maintain context from previous exchanges
- Reference and build upon previous answers when relevant to the current question
- If the user refers to "it", "that", "the paper mentioned before", etc., use the conversation history to understand what they're referring to

Fact Verification & Tables:
- When creating tables or extracting technical details (e.g., sensors, accuracy, participants):
  - Look for EXPLICIT evidence or strong contextual indicators in the text.
  - Be aware of synonyms (e.g., "subjective measures" might be the "ground truth" label).
  - If a paper mentions a sensor in the Introduction as "related work" but does not use it, DO NOT list it as the paper's sensor.
  - If the text does not explicitly state the detail, write "Not Specified" or "N/A". Do not guess.
  - Verify that the detail belongs to the correct paper (check the filename/title in the context).

Formatting Guidelines:
- Use markdown formatting for better readability
- When asked for tables, use proper markdown table syntax:
  | Column1 | Column2 | Column3 |
  |---------|---------|----------|
  | Data    | Data    | Data     |
- For summaries of multiple papers, create structured tables with columns like:
  Paper Title, Authors, Key Findings, Methodology, etc.
- Use bullet points for lists
- Use headers (##) to organize information
- Be precise and academic in tone
- Cite paper names when referencing specific information
- If asked about something not in the papers, acknowledge the limitation
- Provide comprehensive answers while being well-structured

Visualization Guidelines (Mermaid.js):

- Always use Mermaid.js when the user asks for diagrams, flowcharts, mind maps, or graphs.

- Put Mermaid code inside a fenced code block with the language tag `mermaid`:
  ```mermaid
  graph TD
      A[Start] --> B[End]
  ```

- Never use LaTeX syntax inside Mermaid diagrams.
  - Forbidden: $...$, $$...$$, \(...\)
  - Mermaid cannot render LaTeX → the diagram will break.
  - Use plain text alternatives:
    - Instead of $x$ → x
    - Instead of $\lambda$ → lambda or λ
    - Instead of $\sigma^2$ → sigma^2
  - Keep node labels simple text.
  - Avoid formatting, math markup, newlines.
  - If a newline is unavoidable, use <br/>.

- Avoid parentheses and brackets inside node labels.
  - These characters conflict with Mermaid's own syntax.
  - ❌ BAD: A[Function f(x)]
  - ✅ GOOD: A["Function f(x)"]
  - ❌ BAD: B{Proposed Framework: <br/>Simulate Industrial Duties in Fitness Setting}
  - ✅ GOOD: B{"Proposed Framework: <br/>Simulate Industrial Duties in Fitness Setting"}
  - ALWAYS wrap the label in double quotes if it contains spaces or special characters such as (), {}, [ ], :, ->, <, >, etc.
  - If a newline is unavoidable, use <br/> inside a quoted string.

- Do NOT include any commentary, explanation, footnotes, or references inside the mermaid code block.
  - Only raw Mermaid code is allowed.
  - ❌ BAD:
    ```mermaid
    graph TD
    A --> B
    (From Smith et al. 2020)
    ```
  - ✅ GOOD:
    ```mermaid
    graph TD
    A --> B
    ```
    (From Smith et al. 2020)

- If returning multiple diagrams, separate them into multiple code blocks.
  - Mermaid can only render one diagram per code block.

- When outputting both text and Mermaid, always put Mermaid last or clearly separated.

- If the requested diagram cannot be represented cleanly in Mermaid, say so and propose a text-based alternative.
"""


    async def _build_final_prompt_from_orchestration(
        self,
        orchestration: dict,
        prompt_tokens: int,
        response_tokens: int,
        query: str,
        conversation_history: List[Message],
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> tuple[str, List[str], int, int]:
        """Helper to execute worker commands and build the final prompt."""
        relevant_chunks = await self._execute_worker_commands(
            orchestration["commands"],
            allowed_paper_ids=allowed_paper_ids,
            user_query=query,
        )
        if not relevant_chunks:
            raise RuntimeError(
                f"Worker found no chunks for orchestrated commands. "
                f"Strategy: {orchestration.get('strategy')}, Reasoning: {orchestration.get('reasoning')}"
            )
        context = await self._build_context(
            relevant_chunks,
            include_overview=False,
            allowed_paper_ids=allowed_paper_ids,
        )
        source_papers = self._extract_source_papers(relevant_chunks)
        enhanced_query = self._enhance_query(query)
        prompt = self._build_prompt(enhanced_query, context, conversation_history)
        return prompt, source_papers, prompt_tokens, response_tokens

    async def build_prompt_with_orchestration(
        self,
        query: str,
        conversation_history: List[Message],
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> tuple[str, List[str], int, int]:
        """Build a prompt using the orchestrator-worker retrieval flow.

        Args:
            query: User's question
            conversation_history: Previous conversation messages
            paper_id: Optional paper ID to scope retrieval to a specific paper

        Returns a tuple of (prompt, source_papers, prompt_tokens, response_tokens).
        """
        orchestration, prompt_tokens, response_tokens = self._orchestrate_retrieval(
            query,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )
        return await self._build_final_prompt_from_orchestration(
            orchestration, prompt_tokens, response_tokens, query, conversation_history, allowed_paper_ids
        )

    async def build_prompt_with_orchestration_async(
        self,
        query: str,
        conversation_history: List[Message],
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> tuple[str, List[str], int, int]:
        """Build a prompt using the orchestrator-worker retrieval flow asynchronously."""
        orchestration, prompt_tokens, response_tokens = await self._orchestrate_retrieval_async(
            query,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )
        return await self._build_final_prompt_from_orchestration(
            orchestration, prompt_tokens, response_tokens, query, conversation_history, allowed_paper_ids
        )

    async def stream_model_output_async(
        self,
        prompt: str,
        session_id: str,
        pre_prompt_tokens: int = 0,
        pre_response_tokens: int = 0,
    ) -> "AsyncIterator[tuple[str, Optional[TokenUsage]]]":
        """Stream model output for a prepared prompt asynchronously."""
        full_text = ""
        # Streaming via GenerativeModel async
        response = await self.model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            chunk_text = ""
            try:
                chunk_text = chunk.text
            except Exception:
                # Fallback: try to pull text parts manually if .text accessor fails
                chunk_text = self._extract_text_from_response(chunk)

            if chunk_text:
                full_text += chunk_text
                yield chunk_text, None

        # Estimate token usage using count_tokens
        prompt_tokens = await self.count_tokens_async(prompt)
        response_tokens = await self.count_tokens_async(full_text)
        
        # Combine with pre-usage (orchestrator tokens)
        total_prompt = prompt_tokens + pre_prompt_tokens
        total_response = response_tokens + pre_response_tokens
        
        token_usage = TokenUsage(
            session_id=session_id,
            prompt_tokens=total_prompt,
            response_tokens=total_response,
            total_tokens=total_prompt + total_response,
            model=settings.agent.model,
        )
        yield "", token_usage

    async def generate_response_with_planning(
        self,
        query: str,
        conversation_history: List[Message],
        session_id: str,
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[str], TokenUsage]:
        """
        Generate a response using the orchestrator-worker flow (synchronous).
        
        Returns:
            Tuple of (response_text, source_papers, token_usage)
        """
        # Build prompt and get orchestration usage
        prompt, source_papers, prompt_tokens, response_tokens = await self.build_prompt_with_orchestration(
            query,
            conversation_history,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )

        # Generate response
        response = await self.model.generate_content_async(prompt)
        response_text = self._extract_text_from_response(response)

        # Calculate total usage
        gen_prompt_tokens = await self.count_tokens_async(prompt)
        gen_response_tokens = await self.count_tokens_async(response_text)
        
        total_prompt = prompt_tokens + gen_prompt_tokens
        total_response = response_tokens + gen_response_tokens
        
        token_usage = TokenUsage(
            session_id=session_id,
            prompt_tokens=total_prompt,
            response_tokens=total_response,
            total_tokens=total_prompt + total_response,
            model=settings.agent.model,
        )

        return response_text, source_papers, token_usage

    def _enrich_summaries(self, summaries: List[dict]) -> None:
        """Enrich summaries with fresh titles from paper_manager in-place."""
        if self.citation_service:
            for p in summaries:
                paper = self.citation_service.paper_manager.get_paper(p['paper_id'])
                if paper:
                    p['paper_title'] = paper.display_title
