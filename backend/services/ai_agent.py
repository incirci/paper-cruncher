"""AI Agent service using Google Gemini with RAG pipeline."""

from typing import List, Optional, Iterator, Tuple

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
    ) -> Tuple[str, List[str]]:
        """Build an Imagen prompt, leveraging RAG context to guide the visual content."""
        # Reuse orchestrator-worker to gather key chunks
        orchestration = self._orchestrate_retrieval(query, paper_id=paper_id)
        relevant_chunks = self._execute_worker_commands(orchestration['commands'])
        context = self._build_context(relevant_chunks, include_overview=True)
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
    ) -> tuple[str, List[str]]:
        """Build a prompt using the orchestrator-worker retrieval flow.

        Args:
            query: User's question
            conversation_history: Previous conversation messages
            paper_id: Optional paper ID to scope retrieval to a specific paper

        Returns a tuple of (prompt, source_papers).
        """
        orchestration = self._orchestrate_retrieval(query, paper_id=paper_id)
        relevant_chunks = self._execute_worker_commands(orchestration['commands'])
        if not relevant_chunks:
            raise RuntimeError(
                f"Worker found no chunks for orchestrated commands. "
                f"Strategy: {orchestration.get('strategy')}, Reasoning: {orchestration.get('reasoning')}"
            )
        context = self._build_context(relevant_chunks, include_overview=True)
        source_papers = self._extract_source_papers(relevant_chunks)
        enhanced_query = self._enhance_query(query)
        prompt = self._build_prompt(enhanced_query, context, conversation_history)
        return prompt, source_papers

    def stream_model_output(
        self,
        prompt: str,
        session_id: str,
    ) -> "Iterator[tuple[str, Optional[TokenUsage]]]":
        """Stream model output for a prepared prompt.

        Yields tuples of (text_chunk, token_usage). token_usage only present at the end.
        """
        full_text = ""
        # Streaming via GenerativeModel
        stream = self.model.generate_content(prompt, stream=True)
        for chunk in stream:
            if getattr(chunk, "text", None):
                full_text += chunk.text
                yield chunk.text, None

        # Estimate token usage using count_tokens
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(full_text)
        token_usage = TokenUsage(
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=prompt_tokens + response_tokens,
            model=settings.agent.model,
        )
        yield "", token_usage

    def _orchestrate_retrieval(self, query: str, paper_id: Optional[str] = None) -> dict:
        """
        Orchestrator Agent: Analyzes query and issues specific commands for information gathering.
        
        This agent specializes in understanding the big picture and coordinating retrieval.
        It decides WHAT information to fetch and FROM WHERE.
        
        Args:
            query: User's question
            paper_id: Optional paper ID to scope retrieval to a specific paper
            
        Returns:
            Dictionary with orchestration commands
        """
        paper_summaries = self.vector_db.get_paper_summaries()
        
        # Filter to specific paper if paper_id is provided
        if paper_id:
            paper_summaries = [p for p in paper_summaries if p.get('paper_id') == paper_id]
            if not paper_summaries:
                raise RuntimeError(f"Paper with ID {paper_id} not found")
        
        # Build a simple list of paper names only (avoid potentially unsafe content in summaries)
        paper_names = [p['paper_filename'] for p in paper_summaries]
        papers_list = ", ".join(paper_names)
        
        orchestration_prompt = f"""You are a retrieval coordinator. Analyze this question and decide how to gather information.

Available papers: {papers_list}

Question: {query}

Create retrieval commands in this exact format:

COMMAND 1:
ACTION: fetch_summary
PAPERS: ALL
FOCUS: main topics

STRATEGY: consolidate_all
REASONING: Need overview of all papers

Valid actions: fetch_summary (overview), fetch_details (specific search), fetch_comparison (compare papers)
For PAPERS, use: ALL or specific filenames separated by commas
"""
        
        def _call_orchestrator(prompt_text: str, max_output_tokens: int = 300):
            try:
                return self.orchestrator_model.generate_content(prompt_text)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Orchestration failed: {str(e)}")

        # First attempt
        orchestration_response = _call_orchestrator(orchestration_prompt, max_output_tokens= getattr(settings.agent, "orchestrator_max_output_tokens", 300))
        
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
            elif line.startswith('STRATEGY:'):
                strategy = line.split(':', 1)[1].strip().lower()
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        if current_command:
            commands.append(current_command)
        
        return {'commands': commands, 'strategy': strategy, 'reasoning': reasoning}

    def _execute_worker_commands(self, commands: list) -> List[dict]:
        """
        Worker Agent: Executes specific retrieval commands from orchestrator.
        
        This agent specializes in deep dives and extracting specific information.
        """
        all_chunks = []
        chunk_ids_seen = set()
        
        for cmd in commands:
            action = cmd.get('action', '')
            papers = cmd.get('papers', [])
            focus = cmd.get('focus', '')
            
            if not papers:
                continue
            
            all_papers = self.vector_db.get_all_papers()
            # Use canonical title for coordination, but keep filename mapping for backwards compatibility
            paper_id_map = {}
            for p in all_papers:
                filename = p.get('paper_filename')
                title = p.get('paper_title') or filename
                if filename:
                    paper_id_map[filename] = p['paper_id']
                if title:
                    paper_id_map[title] = p['paper_id']
            paper_ids = [paper_id_map[fname] for fname in papers if fname in paper_id_map]
            
            if not paper_ids:
                continue
            
            if action == 'fetch_summary':
                for paper_id in paper_ids:
                    chunks = self.vector_db.get_paper_chunks(paper_id)[:2]
                    for chunk in chunks:
                        if chunk['id'] not in chunk_ids_seen:
                            all_chunks.append(chunk)
                            chunk_ids_seen.add(chunk['id'])
            
            elif action == 'fetch_details':
                search_query = focus if focus else "detailed content"
                chunks = self.vector_db.search(
                    query=search_query,
                    n_results=settings.chunking.max_chunks_per_query * 2,
                    paper_ids=paper_ids,
                )
                for chunk in chunks:
                    if chunk['id'] not in chunk_ids_seen:
                        all_chunks.append(chunk)
                        chunk_ids_seen.add(chunk['id'])
            
            elif action == 'fetch_comparison':
                search_query = focus if focus else "methodology results"
                chunks = self.vector_db.search(
                    query=search_query,
                    n_results=min(len(paper_ids) * 3, 20),
                    paper_ids=paper_ids,
                )
                for chunk in chunks:
                    if chunk['id'] not in chunk_ids_seen:
                        all_chunks.append(chunk)
                        chunk_ids_seen.add(chunk['id'])
        
        return all_chunks

    def generate_response_with_planning(
        self,
        query: str,
        conversation_history: List[Message],
        session_id: str,
    ) -> tuple[str, List[str], TokenUsage]:
        """
        Generate response using two-tier approach: planning then focused retrieval.

        Args:
            query: User's question
            conversation_history: Previous messages in the conversation
            session_id: Current session ID

        Returns:
            Tuple of (response_text, source_papers, token_usage)
        """
        # Step 1: Orchestrator analyzes and issues commands
        orchestration = self._orchestrate_retrieval(query)
        
        # Step 2: Worker executes commands and gathers information
        relevant_chunks = self._execute_worker_commands(orchestration['commands'])
        
        # If no chunks retrieved, raise error
        if not relevant_chunks:
            raise RuntimeError(f"Worker found no chunks for orchestrated commands. Strategy: {orchestration.get('strategy')}, Reasoning: {orchestration.get('reasoning')}")

        # Step 3: Build context and generate final response
        context = self._build_context(relevant_chunks, include_overview=True)
        source_papers = self._extract_source_papers(relevant_chunks)
        enhanced_query = self._enhance_query(query)
        prompt = self._build_prompt(enhanced_query, context, conversation_history)
        
        response = self.model.generate_content(prompt)

        out_text = getattr(response, "text", "") or ""

        # Token usage (best-effort via count_tokens)
        prompt_tokens = self.count_tokens(prompt)
        response_tokens = self.count_tokens(out_text)
        token_usage = TokenUsage(
            session_id=session_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=prompt_tokens + response_tokens,
            model=settings.agent.model,
        )

        return out_text, source_papers, token_usage

    def generate_response_stream(
        self,
        query: str,
        conversation_history: List[Message],
        session_id: str,
    ) -> Iterator[tuple[str, List[str], Optional[TokenUsage]]]:
        """
        Generate streaming response using RAG pipeline.

        Args:
            query: User's question
            conversation_history: Previous messages in the conversation
            session_id: Current session ID

        Yields:
            Tuples of (text_chunk, source_papers, token_usage)
            token_usage is only provided in the final chunk
        """
        # Use the orchestrator-worker flow to prepare prompt
        prompt, source_papers = self.build_prompt_with_orchestration(query, conversation_history)

        # Stream the model output
        for chunk_text, usage in self.stream_model_output(prompt, session_id):
            yield chunk_text, source_papers, usage

    def _enhance_query(self, query: str) -> str:
        """
        Enhance user query to emphasize specific formatting requirements.
        
        This helps the AI model follow user instructions more precisely by
        reinforcing key formatting requirements.
        """
        query_lower = query.lower()
        
        # Detect if user wants a table
        if any(word in query_lower for word in ['table', 'tabular', 'format']):
            if 'lean' in query_lower or 'concise' in query_lower or 'brief' in query_lower or 'short' in query_lower:
                return f"{query}\n\nIMPORTANT: Provide ONLY a concise table with essential columns. No lengthy explanations before or after the table."
            else:
                return f"{query}\n\nIMPORTANT: Format the response as a markdown table."
        
        # Detect if user wants brief/concise response
        if any(word in query_lower for word in ['brief', 'concise', 'short', 'lean', 'quick']):
            return f"{query}\n\nIMPORTANT: Keep the response concise and to the point."
        
        return query

    def _build_context(self, chunks: List[dict], include_overview: bool = False) -> str:
        """Build context string from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks from vector DB
            include_overview: Whether to include paper overview at the top
        """
        if not chunks:
            return "No relevant information found in the papers."

        context_parts = []
        
        # Add paper overview if needed (for broad queries)
        if include_overview:
            papers = self._extract_source_papers(chunks)
            context_parts.append("# Available Papers Overview\n")
            context_parts.append(f"Total papers in context: {len(papers)}\n")
            context_parts.append("Papers: " + ", ".join(papers) + "\n\n")
        
        context_parts.append("# Relevant Information from Papers\n")

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            paper_name = metadata.get("paper_filename", "Unknown")
            page_num = metadata.get("page_number", "N/A")

            context_parts.append(
                f"\n## Source {i}: {paper_name} (Page {page_num})\n{chunk['content']}\n"
            )

        return "\n".join(context_parts)

    def _extract_source_papers(self, chunks: List[dict]) -> List[str]:
        """Extract unique source paper filenames from chunks."""
        papers = set()
        for chunk in chunks:
            paper_name = chunk["metadata"].get("paper_filename")
            if paper_name:
                papers.add(paper_name)
        return sorted(list(papers))

    def _build_prompt(
        self, query: str, context: str, conversation_history: List[Message]
    ) -> str:
        """Build complete prompt with system instructions, context, history, and query."""
        import logging
        logger = logging.getLogger(__name__)
        
        prompt_parts = [self.system_prompt, "\n"]

        # Add conversation history FIRST (before context) so it's more prominent
        if conversation_history:
            logger.info(f"Building prompt with {len(conversation_history)} history messages")
            prompt_parts.append("# Conversation History\n")
            prompt_parts.append("Below is the conversation so far. Use this context to understand references and maintain continuity:\n\n")
            # Only include recent messages to save tokens
            recent_history = conversation_history[-10:]  # Last 10 messages (increased from 5)
            for msg in recent_history:
                role = "User" if msg.role == MessageRole.USER else "Assistant"
                prompt_parts.append(f"**{role}**: {msg.content}\n\n")
            prompt_parts.append("---\n\n")
        else:
            logger.info("Building prompt with NO conversation history")

        # Add context from papers
        prompt_parts.append("# Available Information from Papers\n")
        prompt_parts.append(context)
        prompt_parts.append("\n---\n")

        # Add current query
        prompt_parts.append(f"# Current Question\n{query}\n\n")
        
        if conversation_history:
            prompt_parts.append(
                "Please answer the current question using:\n"
                "1. The conversation history above to understand context and references\n"
                "2. The paper information provided\n"
                "3. Maintain continuity with previous responses when relevant\n"
            )
        else:
            prompt_parts.append(
                "Please provide a comprehensive answer based on the papers above.\n"
            )

        return "\n".join(prompt_parts)

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Use the model's count_tokens where available
        try:
            return int(self.model.count_tokens(text).total_tokens)
        except Exception:
            # Fallback heuristic if count_tokens is unavailable
            return max(1, len(text.split()))
