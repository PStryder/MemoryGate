"""
Minimal import harness for SaaS compatibility checks.

This module must not import app wiring or FastAPI.
"""

import core.context  # noqa: F401
import core.models  # noqa: F401
import core.audit  # noqa: F401
import core.services.memory  # noqa: F401
