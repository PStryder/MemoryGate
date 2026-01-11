"""
Compatibility entrypoint for MemoryGate.
"""

from app.main import app, asgi_app  # noqa: F401
from core.db import DB  # noqa: F401
from core.mcp import mcp_sse_app, mcp_stream_app  # noqa: F401
from core.config import *  # noqa: F401,F403
from core.services.memory_service import *  # noqa: F401,F403
