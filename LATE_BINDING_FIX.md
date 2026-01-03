## Late-Binding Pattern Fix - Final Implementation

### Root Cause Analysis

**Problem:** `SessionLocal` was captured as `None` at module import time, before `init_db()` ran.

**Original broken code:**
```python
# server.py (module level - runs at import time)
SessionLocal = None  # Initial value

# Later, at module level (BEFORE init_db runs)
app.mount("/mcp/", MCPAuthGateASGI(mcp_app, SessionLocal))  # Captures None!

# Much later, during startup
def init_db():
    global SessionLocal
    SessionLocal = sessionmaker(...)  # Too late - already captured as None
```

The `global` statement updates `server.SessionLocal`, but the MCP gate and oauth_routes already captured the `None` value at import time.

---

### Solution: Late-Binding Pattern

**DB Holder Class:**
```python
# server.py
class DB:
    engine = None
    SessionLocal = None

def init_db():
    DB.SessionLocal = sessionmaker(...)  # Updates class attribute
```

**Lazy Evaluation (Lambda):**
```python
# server.py - mount with lambda for late binding
app.mount("/mcp/", MCPAuthGateASGI(mcp_app, lambda: DB.SessionLocal))
```

**MCP Gate (Proper Implementation):**
```python
# mcp_auth_gate.py
class MCPAuthGateASGI:
    def __init__(self, wrapped_app, sessionmaker_getter):
        self.get_sessionmaker = sessionmaker_getter  # Store callable
    
    async def __call__(self, scope, receive, send):
        SessionLocal = self.get_sessionmaker()  # Call lambda to get current value
        if SessionLocal is None:
            await self._send_503(send)  # Fail loudly with 503
            return
        
        db = SessionLocal()  # Create session instance
        try:
            ...
        finally:
            db.close()
```

**OAuth Routes (Safe Import):**
```python
# oauth_routes.py
from server import DB  # Import DB class directly

def get_db_session():
    if DB.SessionLocal is None:
        raise RuntimeError("Database not initialized")
    db = DB.SessionLocal()
    ...
```

---

### Key Concepts

1. **Late Binding:** The lambda `lambda: DB.SessionLocal` doesn't evaluate until called
2. **Proper Naming:** `sessionmaker_getter` / `get_sessionmaker()` (not "factory" or "class")
3. **Sequencing:**
   - `get_sessionmaker()` → returns the sessionmaker callable
   - `SessionLocal()` → creates the per-request DB session
4. **Fail Loudly:** 503 when SessionLocal is None (not silent failures)

---

### Test Sequence

```bash
# 1. Deploy
fly deploy --no-cache

# 2. Health check
curl https://memorygate.fly.dev/health
# Expected: {"status":"ok","service":"memorygate"}

# 3. MCP without auth (should fail)
curl https://memorygate.fly.dev/mcp/
# Expected: 401 {"error":"Unauthorized",...}

# 4. Get API key via client credentials
echo {"client_id":"ZEaaiWOQfw","client_secret":"aEVFvb8tXf_Ppy9WbelF1cutjrvEP3Zn"} > creds.json
curl -X POST https://memorygate.fly.dev/auth/client -H "Content-Type: application/json" -d @creds.json
# Expected: {"api_key":"mg_...","key_prefix":"mg_...","user_id":"...","expires_at":null}

# 5. MCP with auth (should succeed)
curl https://memorygate.fly.dev/mcp/ -H "Authorization: Bearer mg_YOUR_KEY_HERE"
# Expected: 200 or SSE stream

# 6. Verify authenticated user
curl https://memorygate.fly.dev/auth/me -H "Authorization: Bearer mg_YOUR_KEY_HERE"
# Expected: {"id":"...","email":"ZEaaiWOQfw@client.memorygate.internal",...}
```

---

### Failure Modes & Diagnostics

**If #4 fails with 500:**
- Check logs for "Database not initialized" RuntimeError
- Means `DB.SessionLocal` is still None in oauth_routes
- Indicates module import issues (unlikely with direct `from server import DB`)

**If #4 works but #5 fails with 401:**
- MCP gate is working but auth validation failing
- Check header parsing or key verification logic
- Verify API key was stored correctly in database

**If #4 fails with 503:**
- MCP gate sees SessionLocal as None
- Lambda isn't returning the sessionmaker
- Check init_db() ran successfully (look for "Database initialized" in logs)

---

### Why This Works

1. **No early capture:** Lambda defers evaluation until runtime
2. **Class attributes:** `DB.SessionLocal` is a single shared reference
3. **Module safety:** Direct import `from server import DB` avoids module instance issues
4. **Explicit failures:** 503/RuntimeError instead of silent NoneType crashes

This pattern is production-safe and handles initialization ordering correctly.

## Module Duality Fix - The Final Dragon

### The Problem

Even with the `DB` class holder, you can still get separate module instances if the app starts in a way that creates a `__main__` module:

**Broken pattern:**
```bash
# Dockerfile
CMD ["python", "server.py"]

# This creates two module instances:
# 1. __main__ (from running server.py as a script)
# 2. server (when other modules import it)

# Result: Two separate DB class instances!
# from server import DB  # Different DB than __main__.DB
```

When `oauth_routes.py` does `from server import DB`, it gets the `server` module instance. But if the app was started with `python server.py`, `init_db()` ran in the `__main__` module instance, updating `__main__.DB.SessionLocal` instead of `server.DB.SessionLocal`.

**Two module instances = Two DB instances = SessionLocal still None**

### The Solution: Module-Level ASGI App

**Expose ASGI app at module level:**
```python
# server.py (module level, not in __main__)
asgi_app = SlashNormalizerASGI(app)

# Keep __main__ block for local dev convenience
if __name__ == "__main__":
    print("MemoryGate starting...")
    uvicorn.run(asgi_app, host="0.0.0.0", port=8080)
```

**Start via uvicorn pointing to module symbol:**
```dockerfile
# Dockerfile
CMD ["uvicorn", "server:asgi_app", "--host", "0.0.0.0", "--port", "8080"]
```

**Why this works:**
- `uvicorn server:asgi_app` imports `server` as a module (not `__main__`)
- All `from server import DB` statements get the same module instance
- `init_db()` and `get_db_session()` both see the same `DB.SessionLocal`
- No ghost modules

### Production vs Development

**Production (Fly.io):**
```bash
uvicorn server:asgi_app --host 0.0.0.0 --port 8080
# Uses server module instance everywhere
```

**Local Development:**
```bash
python server.py
# Still works! __main__ block calls uvicorn.run(asgi_app)
# asgi_app is defined at module level, so it works the same way
```

### Verification

**After deployment, check logs for:**
```
INFO:memorygate:Database initialized
INFO: Uvicorn running on http://0.0.0.0:8080
```

**Test module consistency:**
```bash
# This should work now (was failing before)
curl -X POST https://memorygate.fly.dev/auth/client \
  -H "Content-Type: application/json" \
  -d '{"client_id":"ZEaaiWOQfw","client_secret":"aEVFvb8tXf_Ppy9WbelF1cutjrvEP3Zn"}'
```

If it still fails with "Database not initialized", there's a deeper module loading issue. But with `uvicorn server:asgi_app`, this should be impossible.

### Summary

**Before:**
- Dockerfile: `CMD ["python", "server.py"]`
- Creates `__main__` module
- `from server import DB` gets different instance
- SessionLocal is None

**After:**
- Dockerfile: `CMD ["uvicorn", "server:asgi_app", ...]`
- Only `server` module exists
- All imports get same instance
- SessionLocal is properly initialized

**This is the bulletproof pattern.**
