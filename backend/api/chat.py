"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
import json
import logging

from backend.models.schemas import ChatRequest, ChatResponse, MessageRole, Conversation

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_session_context(app_state, session_id: str) -> tuple[list[str], str | None]:
    """Return (paper_ids, selected_paper_id) for a session.

    Always returns a concrete list for paper_ids (possibly empty) so downstream
    code never needs to infer or fall back to global papers.
    """
    conversation = app_state.conversation_manager.get_conversation(session_id)
    if not conversation:
        return [], None
    return conversation.paper_ids or [], conversation.selected_paper_id


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """Send a message to the AI agent and get a response."""

    app_state = request.app.state.app_state

    # Create or validate session
    if chat_request.session_id:
        if not app_state.conversation_manager.session_exists(chat_request.session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = chat_request.session_id
    else:
        session_id = app_state.conversation_manager.create_session()

    # Add user message to conversation
    app_state.conversation_manager.add_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=chat_request.message,
    )

    # Get conversation history
    history = app_state.conversation_manager.get_conversation_history(session_id)

    # Load session paper context so retrieval is strictly scoped to this session
    paper_ids, selected_paper_id = _get_session_context(app_state, session_id)
    allowed_paper_ids: list[str] = paper_ids

    # If no papers are registered for this session, do not attempt RAG.
    # This prevents accidental fallback to all papers in the vector DB.
    if not allowed_paper_ids:
        raise HTTPException(
            status_code=400,
            detail="No papers are registered for this session yet. Add papers to the session before asking questions.",
        )

    try:
        logger.info(
            "CHAT /chat session_id=%s allowed_paper_ids=%s selected_paper_id=%s",
            session_id,
            allowed_paper_ids,
            selected_paper_id,
        )
    except Exception:
        pass

    # Visualization branch
    if request.app.state.settings.image.enabled and app_state.ai_agent.is_visualization_request(chat_request.message):
        prompt, source_papers = app_state.ai_agent._build_image_prompt(
            query=chat_request.message,
            conversation_history=history[:-1],
            paper_id=getattr(chat_request, "paper_id", None),
            allowed_paper_ids=allowed_paper_ids,
        )
        mime = request.app.state.settings.image.mime_type
        width = request.app.state.settings.image.width
        height = request.app.state.settings.image.height
        try:
            mime_type, b64 = app_state.ai_agent.generate_image_bytes(
                prompt=prompt,
                mime_type=mime,
                width=width,
                height=height,
            )
            data_url = f"data:{mime_type};base64,{b64}"
            app_state.conversation_manager.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content="(Generated image)",
                token_count=0,
                source_papers=source_papers,
            )
            return ChatResponse(
                session_id=session_id,
                message=f"![Generated visualization]({data_url})",
                source_papers=source_papers,
                token_usage=None,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    # Normal RAG response
    response_text, source_papers, token_usage = app_state.ai_agent.generate_response_with_planning(
        query=chat_request.message,
        conversation_history=history[:-1],  # Exclude the just-added user message
        session_id=session_id,
        paper_id=chat_request.paper_id,
        allowed_paper_ids=allowed_paper_ids,
    )

    app_state.token_tracker.record_usage(token_usage)

    app_state.conversation_manager.add_message(
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=response_text,
        token_count=token_usage.total_tokens,
        source_papers=source_papers,
    )

    return ChatResponse(
        session_id=session_id,
        message=response_text,
        source_papers=source_papers,
        token_usage=token_usage,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = Form(..., description="User message to send to the AI agent"),
    session_id: str | None = Form(None, description="Existing session ID to continue"),
    paper_id: str | None = Form(None, description="Optional paper ID to scope retrieval"),
    files: list[UploadFile] | None = File(None, description="Optional uploaded files (images/docs)")
):
    """
    Send a message to the AI agent and get a streaming response.

    Args:
        chat_request: Chat request with message and optional session_id

    Returns:
        Server-sent events stream with AI response chunks
    """
    app_state = request.app.state.app_state

    # Create or validate session
    if session_id:
        if not app_state.conversation_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session_id = app_state.conversation_manager.create_session()

    # Add user message to conversation
    app_state.conversation_manager.add_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=message,
    )

    # Get conversation history
    history = app_state.conversation_manager.get_conversation_history(session_id)

    # Load session paper context for session-scoped retrieval
    paper_ids, selected_paper_id = _get_session_context(app_state, session_id)
    allowed_paper_ids: list[str] = paper_ids

    # Block RAG when the session has no paper context.
    if not allowed_paper_ids:
        async def empty_stream():
            msg = {
                "type": "error",
                "message": "No papers are registered for this session yet. Add papers to the session before asking questions.",
            }
            yield f"data: {json.dumps(msg)}\n\n"

        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        logger.info(
            "CHAT /chat/stream (POST) session_id=%s allowed_paper_ids=%s selected_paper_id=%s",
            session_id,
            allowed_paper_ids,
            selected_paper_id,
        )
    except Exception:
        pass

    async def generate_stream():
        """Generate server-sent events stream."""
        import logging
        logger = logging.getLogger(__name__)
        
        full_response = ""
        source_papers = []
        token_usage = None

        try:
            # Send session_id first
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            # Log history info
            logger.info(f"Streaming chat - Session {session_id} has {len(history)} messages in history")
            logger.info(f"Passing {len(history[:-1])} messages to AI agent (excluding current user message)")

            # If visualization requested and enabled, stream an image event first
            if request.app.state.settings.image.enabled and app_state.ai_agent.is_visualization_request(message):
                prompt_img, source_papers = app_state.ai_agent._build_image_prompt(
                    query=message,
                    conversation_history=history[:-1],
                    paper_id=paper_id,
                    allowed_paper_ids=allowed_paper_ids,
                )
                mime = request.app.state.settings.image.mime_type
                width = request.app.state.settings.image.width
                height = request.app.state.settings.image.height
                try:
                    mime_type, b64 = app_state.ai_agent.generate_image_bytes(
                        prompt=prompt_img,
                        mime_type=mime,
                        width=width,
                        height=height,
                    )
                    data_url = f"data:{mime_type};base64,{b64}"
                    yield f"data: {json.dumps({'type': 'image', 'mime_type': mime_type, 'data_url': data_url})}\n\n"
                    # No token usage tracking available for Imagen via this method
                    app_state.conversation_manager.add_message(
                        session_id=session_id,
                        role=MessageRole.ASSISTANT,
                        content="(Generated image)",
                        token_count=0,
                        source_papers=source_papers,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'source_papers': source_papers, 'token_usage': None})}\n\n"
                    return
                except Exception as e:
                    error_msg = f"Image generation failed: {str(e)}"
                    yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                    return

            # Orchestrate and execute retrieval to build prompt, then stream model output
            prompt, source_papers = app_state.ai_agent.build_prompt_with_orchestration(
                query=message,
                conversation_history=history[:-1],
                paper_id=paper_id,
                allowed_paper_ids=allowed_paper_ids,
            )

            # Stream model output
            for chunk_text, usage in app_state.ai_agent.stream_model_output(prompt, session_id):
                if chunk_text:
                    full_response += chunk_text
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk_text})}\n\n"
                if usage:
                    token_usage = usage

            # Record token usage
            if token_usage:
                app_state.token_tracker.record_usage(token_usage)

            # Add assistant message to conversation
            app_state.conversation_manager.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                token_count=token_usage.total_tokens if token_usage else 0,
                source_papers=source_papers,
            )

            # Send completion event with metadata
            token_usage_dict = None
            if token_usage:
                token_usage_dict = token_usage.model_dump()
                # Convert datetime to ISO string for JSON serialization
                if 'timestamp' in token_usage_dict:
                    token_usage_dict['timestamp'] = token_usage_dict['timestamp'].isoformat()
            
            yield f"data: {json.dumps({'type': 'done', 'source_papers': source_papers, 'token_usage': token_usage_dict})}\n\n"
        
        except Exception as e:
            # Send error event
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# GET-based SSE endpoint for EventSource clients
@router.get("/chat/stream")
async def chat_stream_get(
    request: Request,
    message: str = Query(..., description="User message to send to the AI agent"),
    session_id: str | None = Query(None, description="Existing session ID to continue"),
    paper_id: str | None = Query(None, description="Optional paper ID to scope retrieval"),
):
    """
    Streaming chat via GET for native EventSource support.

    Returns server-sent events with the same payload schema as the POST stream endpoint.
    """
    app_state = request.app.state.app_state

    # Create or validate session
    if session_id:
        if not app_state.conversation_manager.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session_id = app_state.conversation_manager.create_session()

    # Add user message to conversation
    app_state.conversation_manager.add_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=message,
    )

    history = app_state.conversation_manager.get_conversation_history(session_id)

    # Load session paper context for session-scoped retrieval
    paper_ids, selected_paper_id = _get_session_context(app_state, session_id)
    allowed_paper_ids: list[str] = paper_ids

    # Block RAG when the session has no paper context.
    if not allowed_paper_ids:
        async def empty_stream():
            import json as _json
            msg = {
                "type": "error",
                "message": "No papers are registered for this session yet. Add papers to the session before asking questions.",
            }
            yield f"data: {_json.dumps(msg)}\n\n"

        return StreamingResponse(
            empty_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def generate_stream():
        import json as _json
        from asyncio import sleep
        full_response = ""
        source_papers: list[str] = []
        token_usage = None

        # Send session immediately (helps flush proxies)
        yield f"data: {_json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        try:
            # Visualization branch (Imagen) for EventSource
            if request.app.state.settings.image.enabled and app_state.ai_agent.is_visualization_request(message):
                prompt_img, source_papers = app_state.ai_agent._build_image_prompt(
                    query=message,
                    conversation_history=history[:-1],
                    paper_id=paper_id,
                    allowed_paper_ids=allowed_paper_ids,
                )
                mime = request.app.state.settings.image.mime_type
                width = request.app.state.settings.image.width
                height = request.app.state.settings.image.height
                try:
                    mime_type, b64 = app_state.ai_agent.generate_image_bytes(
                        prompt=prompt_img,
                        mime_type=mime,
                        width=width,
                        height=height,
                    )
                    data_url = f"data:{mime_type};base64,{b64}"
                    yield f"data: {json.dumps({'type': 'image', 'mime_type': mime_type, 'data_url': data_url})}\n\n"
                    app_state.conversation_manager.add_message(
                        session_id=session_id,
                        role=MessageRole.ASSISTANT,
                        content="(Generated image)",
                        token_count=0,
                        source_papers=source_papers,
                    )
                    yield f"data: {json.dumps({'type': 'done', 'source_papers': source_papers, 'token_usage': None})}\n\n"
                    return
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Image generation failed: {str(e)}'})}\n\n"
                    return
            # Build prompt with retrieval orchestration
            prompt, source_papers = app_state.ai_agent.build_prompt_with_orchestration(
                query=message,
                conversation_history=history[:-1],
                paper_id=paper_id,
                allowed_paper_ids=allowed_paper_ids,
            )

            # Stream the model output
            for chunk_text, usage in app_state.ai_agent.stream_model_output(prompt, session_id):
                if chunk_text:
                    full_response += chunk_text
                    yield f"data: {_json.dumps({'type': 'chunk', 'text': chunk_text})}\n\n"
                if usage:
                    token_usage = usage

            if token_usage:
                app_state.token_tracker.record_usage(token_usage)

            app_state.conversation_manager.add_message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=full_response,
                token_count=token_usage.total_tokens if token_usage else 0,
                source_papers=source_papers,
            )

            token_usage_dict = None
            if token_usage:
                token_usage_dict = token_usage.model_dump()
                if 'timestamp' in token_usage_dict:
                    token_usage_dict['timestamp'] = token_usage_dict['timestamp'].isoformat()

            yield f"data: {_json.dumps({'type': 'done', 'source_papers': source_papers, 'token_usage': token_usage_dict})}\n\n"

        except Exception as e:  # noqa: BLE001 - send error downstream
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/chat/history/{session_id}")
async def get_history(request: Request, session_id: str):
    """Get conversation history for a session."""
    app_state = request.app.state.app_state

    conversation = app_state.conversation_manager.get_conversation(session_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")

    return conversation


@router.delete("/chat/history/{session_id}")
async def delete_history(request: Request, session_id: str):
    """Delete conversation history for a session."""
    app_state = request.app.state.app_state

    if not app_state.conversation_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    app_state.conversation_manager.delete_conversation(session_id)
    app_state.token_tracker.delete_session_usage(session_id)

    return {"message": "Conversation deleted successfully"}


@router.get("/chat/sessions")
async def list_sessions(request: Request):
    """List all conversation sessions that have messages."""
    app_state = request.app.state.app_state
    sessions = app_state.conversation_manager.list_sessions()

    # Only return sessions with messages
    kept_sessions: list[dict] = []
    for sess in sessions:
        message_count = sess.get("message_count", 0)
        if message_count > 0:
            kept_sessions.append(sess)

    return {"sessions": kept_sessions}


@router.post("/chat/session")
async def create_session(request: Request):
    """Create a new empty chat session and return its ID.

    This is useful for pre-creating a session when a new window is opened so
    that any papers uploaded in that window can be immediately attached to
    the session before the first question is asked.
    """
    app_state = request.app.state.app_state
    session_id = app_state.conversation_manager.create_session()
    return {"session_id": session_id}


@router.delete("/chat/session/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a specific chat session and its token usage.

    Useful when the user wants to discard a session entirely and start
    fresh without leaving stale sessions in history.
    """
    app_state = request.app.state.app_state

    if not app_state.conversation_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    app_state.conversation_manager.delete_conversation(session_id)
    app_state.token_tracker.delete_session_usage(session_id)

    return {"message": "Session deleted"}


@router.get("/chat/debug/sessions")
async def debug_sessions(request: Request):
    """Debug endpoint: return all sessions with paper context and last source_papers.

    This is intended for development/debugging only. It surfaces, for each session:
    - session_id
    - paper_ids
    - selected_paper_id
    - last assistant message's source_papers (if any)
    """
    app_state = request.app.state.app_state

    # Base session info
    sessions = app_state.conversation_manager.list_sessions()
    debug_payload: list[dict] = []

    for sess in sessions:
        # list_sessions currently returns plain dicts
        session_id = sess.get("session_id") if isinstance(sess, dict) else getattr(sess, "session_id", None)
        if not session_id:
            continue

        conv = app_state.conversation_manager.get_conversation(session_id)
        paper_ids = conv.paper_ids if conv else []
        selected_paper_id = conv.selected_paper_id if conv else None

        # Find last assistant message and its source_papers
        history = app_state.conversation_manager.get_conversation_history(session_id)
        last_source_papers: list[str] = []
        if history:
            for msg in reversed(history):
                if getattr(msg, "role", None) == MessageRole.ASSISTANT:
                    last_source_papers = getattr(msg, "source_papers", []) or []
                    break

        debug_payload.append(
            {
                "session_id": session_id,
                "paper_ids": paper_ids or [],
                "selected_paper_id": selected_paper_id,
                "last_source_papers": last_source_papers,
            }
        )

    return {"sessions": debug_payload}


@router.post("/chat/session/{session_id}/context")
async def update_session_context(
    request: Request,
    session_id: str,
    payload: dict = Body(...),
):
    """Update paper context (selected paper and paper set) for a session.

    This allows the frontend to persist which papers were available and which
    one was selected when saving or continuing a session.
    """
    # Extract expected fields from JSON payload
    selected_paper_id = payload.get("selected_paper_id")
    paper_ids = payload.get("paper_ids") or []

    # Enforce invariant: selected_paper_id must be part of paper_ids (or None)
    if selected_paper_id and selected_paper_id not in paper_ids:
        selected_paper_id = None

    app_state = request.app.state.app_state

    if not app_state.conversation_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch current conversation to compare paper_ids. If the paper set
    # changes, we reset the chat history so that what the user sees in
    # the UI always matches the session's paper context and there are
    # no stale answers tied to an older paper set.
    conversation = app_state.conversation_manager.get_conversation(session_id)
    old_paper_ids = conversation.paper_ids if conversation else []

    app_state.conversation_manager.update_session_papers(
        session_id=session_id,
        selected_paper_id=selected_paper_id,
        paper_ids=paper_ids,
    )

    # If the set of papers changed (not just order), clear history and
    # token usage for this session. This keeps session state and UI
    # aligned when papers are added/removed/replaced.
    if set(old_paper_ids or []) != set(paper_ids or []):
        app_state.conversation_manager.delete_messages(session_id)
        app_state.token_tracker.delete_session_usage(session_id)
    # Return the canonical state so the frontend can always sync its
    # local view (currentPaperIds, selectedPaperId) to whatever the
    # backend actually stored after validation/deduplication.
    updated = app_state.conversation_manager.get_conversation(session_id)

    return {
        "message": "Session context updated",
        "paper_ids": updated.paper_ids if updated else paper_ids,
        "selected_paper_id": updated.selected_paper_id if updated else selected_paper_id,
    }


@router.get("/chat/session/{session_id}", response_model=Conversation)
async def get_session_with_context(request: Request, session_id: str):
    """Get a full session including messages and paper context."""
    app_state = request.app.state.app_state
    conversation = app_state.conversation_manager.get_conversation(session_id)

    if not conversation:
        raise HTTPException(status_code=404, detail="Session not found")

    return conversation
