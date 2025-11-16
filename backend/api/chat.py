"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
import json

from backend.models.schemas import ChatRequest, ChatResponse, MessageRole

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Send a message to the AI agent and get a response.

    Args:
        chat_request: Chat request with message and optional session_id

    Returns:
        Chat response with AI message and metadata
    """
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

    # If visualization requested and enabled, delegate to Imagen and embed as markdown image
    if request.app.state.settings.image.enabled and app_state.ai_agent.is_visualization_request(chat_request.message):
        # Build image prompt using RAG context
        prompt, source_papers = app_state.ai_agent._build_image_prompt(
            query=chat_request.message,
            conversation_history=history[:-1],
            paper_id=getattr(chat_request, 'paper_id', None),
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
            # Store a lightweight assistant record without embedding the whole image
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

    # Generate response using orchestrator-worker flow (single architecture)
    response_text, source_papers, token_usage = app_state.ai_agent.generate_response_with_planning(
        query=chat_request.message,
        conversation_history=history[:-1],  # Exclude the just-added user message
        session_id=session_id,
    )

    # Record token usage
    app_state.token_tracker.record_usage(token_usage)

    # Add assistant message to conversation
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
async def chat_stream(request: Request, chat_request: ChatRequest):
    """
    Send a message to the AI agent and get a streaming response.

    Args:
        chat_request: Chat request with message and optional session_id

    Returns:
        Server-sent events stream with AI response chunks
    """
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
            if request.app.state.settings.image.enabled and app_state.ai_agent.is_visualization_request(chat_request.message):
                prompt_img, source_papers = app_state.ai_agent._build_image_prompt(
                    query=chat_request.message,
                    conversation_history=history[:-1],
                    paper_id=getattr(chat_request, 'paper_id', None),
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
                query=chat_request.message,
                conversation_history=history[:-1],
                paper_id=chat_request.paper_id,
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
    """List all conversation sessions."""
    app_state = request.app.state.app_state
    sessions = app_state.conversation_manager.list_sessions()
    return {"sessions": sessions}
