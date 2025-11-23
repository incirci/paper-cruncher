"""AI Agent service using Google Gemini with RAG pipeline."""

from typing import List, Optional, Iterator, Tuple, AsyncIterator

import logging
import google.generativeai as genai

from backend.core.config import settings
from backend.models.schemas import Message, MessageRole, TokenUsage
from backend.services.vector_db import VectorDBService


class AIAgent:
    """AI Agent for answering questions about papers using RAG."""

    def __init__(self, vector_db: VectorDBService):
        """
        Initialize AI Agent with Gemini.

        Args:
            vector_db: Vector database service for semantic search
        """
        self.vector_db = vector_db

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

    def _orchestrate_retrieval(
        self,
        query: str,
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, int, int]:
        """
        Synchronous Orchestrator Agent: Analyzes query and issues specific commands for information gathering.
        
        Args:
            query: User's question
            paper_id: Optional paper ID to scope retrieval to a specific paper
            allowed_paper_ids: Optional list of paper IDs allowed for this session
            
        Returns:
            Tuple of (orchestration commands dict, prompt_tokens, response_tokens)
        """
        paper_summaries = self.vector_db.get_paper_summaries()

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
        
        # Build a simple list of paper names only (avoid potentially unsafe content in summaries)
        paper_names = [p['paper_filename'] for p in paper_summaries]
        papers_list = ", ".join(paper_names)
        
        orchestration_prompt = f'''You are a retrieval coordinator. Analyze this question and decide how to gather information.

Available papers: {papers_list}

Question: {query}

Create retrieval commands in this exact format:

COMMAND 1:
ACTION: fetch_summary
PAPERS: ALL
FOCUS: main topics
DENSITY: normal

STRATEGY: consolidate_all
REASONING: Need overview of all papers

Valid actions: 
1. fetch_summary: Use for high-level overviews, "list all papers", or general summaries. Ignores FOCUS.
2. fetch_details: Use for specific questions where the answer might be in just a few papers. (e.g. "Find the paper that uses LSTM")
3. fetch_comparison: Use for synthesizing specific information across multiple papers. (e.g. "Compare accuracy across all papers", "List the limitations of papers A and B", "Trace the evolution of X"). This searches for the FOCUS topic in each target paper.

For PAPERS, use: ALL or specific filenames separated by commas
For DENSITY, use: normal (default, 5 chunks) or high (deep dive, 20 chunks)
'''
        
        try:
            orchestration_response = self.orchestrator_model.generate_content(
                orchestration_prompt, 
                generation_config={"max_output_tokens": getattr(settings.agent, "orchestrator_max_output_tokens", 300)}
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Orchestration failed: {str(e)}")

        # Check if response was blocked or has issues
        if not getattr(orchestration_response, "candidates", None):
            raise RuntimeError("Orchestration blocked: no candidates returned.")
        
        # Check finish reason
        candidate = orchestration_response.candidates[0]
        
        # Extract text without relying on quick accessor to avoid ValueError
        response_text = ""
        try:
            content = candidate.content
            if content and getattr(content, "parts", None):
                texts: List[str] = []
                for part in content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
                response_text = "\n".join(texts).strip()
        except Exception:
            response_text = ""
        if not response_text:
            raise RuntimeError("Orchestration produced empty content after successful completion.")
        
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
            elif line.startswith('DENSITY:'):
                current_command['density'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('STRATEGY:'):
                strategy = line.split(':', 1)[1].strip().lower()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if current_command:
            commands.append(current_command)
        
        # Calculate tokens
        prompt_tokens = self.count_tokens(orchestration_prompt)
        response_tokens = self.count_tokens(response_text)
        
        return {'commands': commands, 'strategy': strategy, 'reasoning': reasoning}, prompt_tokens, response_tokens

    async def _orchestrate_retrieval_async(
        self,
        query: str,
        paper_id: Optional[str] = None,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[dict, int, int]:
        """
        Async Orchestrator Agent: Analyzes query and issues specific commands for information gathering.
        
        This agent specializes in understanding the big picture and coordinating retrieval.
        It decides WHAT information to fetch and FROM WHERE.
        
        Args:
            query: User's question
            paper_id: Optional paper ID to scope retrieval to a specific paper
            allowed_paper_ids: Optional list of paper IDs allowed for this session
            
        Returns:
            Tuple of (orchestration commands dict, prompt_tokens, response_tokens)
        """
        paper_summaries = self.vector_db.get_paper_summaries()

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
        
        # Build a simple list of paper names only (avoid potentially unsafe content in summaries)
        paper_names = [p['paper_filename'] for p in paper_summaries]
        papers_list = ", ".join(paper_names)
        
        orchestration_prompt = f'''You are a retrieval coordinator. Analyze this question and decide how to gather information.

Available papers: {papers_list}

Question: {query}

Create retrieval commands in this exact format:

COMMAND 1:
ACTION: fetch_summary
PAPERS: ALL
FOCUS: main topics
DENSITY: normal

STRATEGY: consolidate_all
REASONING: Need overview of all papers

Valid actions: 
1. fetch_summary: Use for high-level overviews, "list all papers", or general summaries. Ignores FOCUS.
2. fetch_details: Use for specific questions where the answer might be in just a few papers. (e.g. "Find the paper that uses LSTM")
3. fetch_comparison: Use for synthesizing specific information across multiple papers. (e.g. "Compare accuracy across all papers", "List the limitations of papers A and B", "Trace the evolution of X"). This searches for the FOCUS topic in each target paper.

For PAPERS, use: ALL or specific filenames separated by commas
For DENSITY, use: normal (default, 5 chunks) or high (deep dive, 20 chunks)
'''
        
        try:
            orchestration_response = await self.orchestrator_model.generate_content_async(
                orchestration_prompt, 
                generation_config={"max_output_tokens": getattr(settings.agent, "orchestrator_max_output_tokens", 300)}
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Orchestration failed: {str(e)}")

        # Check if response was blocked or has issues
        if not getattr(orchestration_response, "candidates", None):
            raise RuntimeError("Orchestration blocked: no candidates returned.")
        
        # Check finish reason
        candidate = orchestration_response.candidates[0]
        
        # Extract text without relying on quick accessor to avoid ValueError
        response_text = ""
        try:
            content = candidate.content
            if content and getattr(content, "parts", None):
                texts: List[str] = []
                for part in content.parts:
                    if getattr(part, "text", None):
                        texts.append(part.text)
                response_text = "\n".join(texts).strip()
        except Exception:
            response_text = ""
        if not response_text:
            raise RuntimeError("Orchestration produced empty content after successful completion.")
        
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
            elif line.startswith('DENSITY:'):
                current_command['density'] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('STRATEGY:'):
                strategy = line.split(':', 1)[1].strip().lower()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if current_command:
            commands.append(current_command)
        
        # Calculate tokens
        prompt_tokens = self.count_tokens(orchestration_prompt)
        response_tokens = self.count_tokens(response_text)
        
        return {'commands': commands, 'strategy': strategy, 'reasoning': reasoning}, prompt_tokens, response_tokens

    def _execute_worker_commands(
        self,
        commands: List[dict],
        allowed_paper_ids: Optional[List[str]] = None,
        user_query: str = "",
    ) -> List[dict]:
        """Execute retrieval commands against the vector DB."""
        all_chunks = []
        
        # Get map of filename -> paper_id
        papers = self.vector_db.get_all_papers()
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
                summaries = self.vector_db.get_paper_summaries(paper_ids=effective_ids)
                
                for s in summaries:
                    all_chunks.append({
                        "id": f"summary_{s['paper_id']}",
                        "content": f"Micro-Summary of {s['paper_title']}:\n{s['summary']}",
                        "metadata": {
                            "paper_id": s['paper_id'],
                            "paper_title": s['paper_title'],
                            "paper_filename": s['paper_filename'],
                            "source": "micro_summary"
                        }
                    })
                continue

            # Handle comparison requests: ensure distribution across papers
            # Instead of a global top-N search, we search each paper individually to ensure coverage.
            if action == 'fetch_comparison' and target_ids:
                # Determine chunks per paper. 
                # If density is high, take more. If normal, take fewer to avoid context explosion.
                chunks_per_paper = 5 if density == 'high' else 3
                
                for pid in target_ids:
                    results = self.vector_db.search(
                        query=f"{focus} {user_query}",
                        n_results=chunks_per_paper,
                        paper_ids=[pid]
                    )
                    all_chunks.extend(results)
                continue

            # Construct search query
            search_query = f"{focus} {user_query}"
            
            # Determine n_results based on density
            base_chunks = settings.chunking.max_chunks_per_query
            n_results = (base_chunks * 4) if density == 'high' else base_chunks
            
            # Execute search
            results = self.vector_db.search(
                query=search_query,
                n_results=n_results,
                paper_ids=target_ids if target_ids else allowed_paper_ids
            )
            all_chunks.extend(results)
            
        # Deduplicate chunks by ID
        seen_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk['id'] not in seen_ids:
                seen_ids.add(chunk['id'])
                unique_chunks.append(chunk)
                
        return unique_chunks

    def _build_context(
        self,
        chunks: List[dict],
        include_overview: bool = True,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> str:
        """Format chunks into a context string."""
        context_parts = []

        if include_overview:
            # Add list of available papers to context so the model knows what's in the library
            # regardless of what chunks were retrieved.
            papers = self.vector_db.get_all_papers()
            if allowed_paper_ids:
                papers = [p for p in papers if p['paper_id'] in allowed_paper_ids]
            
            if papers:
                context_parts.append("Available Papers in Knowledge Base:")
                for p in papers:
                    title = p.get('paper_title', p.get('paper_filename', 'Unknown'))
                    context_parts.append(f"- {title}")
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
            title = meta.get('paper_title', meta.get('paper_filename', 'Unknown Paper'))
            
            context_parts.append(f"--- Paper: {title} ---")
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
                history_text += f"{role}: {msg.content}\n"
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

    # --- Imagen integration helpers ---
    def is_visualization_request(self, query: str) -> bool:
        """Heuristic to detect if the user asks for a visual/plot/diagram/image."""
        q = (query or "").lower()
        keywords = [
            "plot", "chart", "graph", "diagram", "visualize", "visualisation", "visualization",
            "draw", "sketch", "figure", "image", "picture", "illustration", "mind map", "mindmap"
        ]
        return any(k in q for k in keywords)

    def _build_image_prompt(
        self,
        query: str,
        conversation_history: List[Message],
        paper_id: Optional[str] = None,
        max_context_chars: int = 2000,
        allowed_paper_ids: Optional[List[str]] = None,
    ) -> Tuple[str, List[str]]:
        """Build an Imagen prompt, leveraging RAG context to guide the visual content."""
        # Reuse orchestrator-worker to gather key chunks, respecting session paper scope
        orchestration, _, _ = self._orchestrate_retrieval(
            query,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )
        relevant_chunks = self._execute_worker_commands(
            orchestration["commands"],
            allowed_paper_ids=allowed_paper_ids,
            user_query=query,
        )
        context = self._build_context(
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

    def generate_image_bytes(
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
            response = model.generate_content(full_prompt)
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
"""


    def build_prompt_with_orchestration(
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
        relevant_chunks = self._execute_worker_commands(
            orchestration["commands"],
            allowed_paper_ids=allowed_paper_ids,
            user_query=query,
        )
        if not relevant_chunks:
            raise RuntimeError(
                f"Worker found no chunks for orchestrated commands. "
                f"Strategy: {orchestration.get('strategy')}, Reasoning: {orchestration.get('reasoning')}"
            )
        context = self._build_context(
            relevant_chunks,
            include_overview=True,
            allowed_paper_ids=allowed_paper_ids,
        )
        source_papers = self._extract_source_papers(relevant_chunks)
        enhanced_query = self._enhance_query(query)
        prompt = self._build_prompt(enhanced_query, context, conversation_history)
        return prompt, source_papers, prompt_tokens, response_tokens

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
        relevant_chunks = self._execute_worker_commands(
            orchestration["commands"],
            allowed_paper_ids=allowed_paper_ids,
            user_query=query,
        )
        if not relevant_chunks:
            raise RuntimeError(
                f"Worker found no chunks for orchestrated commands. "
                f"Strategy: {orchestration.get('strategy')}, Reasoning: {orchestration.get('reasoning')}"
            )
        context = self._build_context(
            relevant_chunks,
            include_overview=True,
            allowed_paper_ids=allowed_paper_ids,
        )
        source_papers = self._extract_source_papers(relevant_chunks)
        enhanced_query = self._enhance_query(query)
        prompt = self._build_prompt(enhanced_query, context, conversation_history)
        return prompt, source_papers, prompt_tokens, response_tokens

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
            # Prefer the SDK's quick accessor when available
            chunk_text = getattr(chunk, "text", None)
            if not chunk_text:
                # Fallback: try to pull text parts manually
                chunk_text = self._extract_text_from_response(chunk)

            if chunk_text:
                full_text += chunk_text
                yield chunk_text, None

        # Estimate token usage using count_tokens
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(full_text)
        
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

    def generate_response_with_planning(
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
        prompt, source_papers, prompt_tokens, response_tokens = self.build_prompt_with_orchestration(
            query,
            conversation_history,
            paper_id=paper_id,
            allowed_paper_ids=allowed_paper_ids,
        )

        # Generate response
        response = self.model.generate_content(prompt)
        response_text = self._extract_text_from_response(response)

        # Calculate total usage
        gen_prompt_tokens = self.count_tokens(prompt)
        gen_response_tokens = self.count_tokens(response_text)
        
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
