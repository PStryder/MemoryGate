# MemoryGate 🧠🔁

**Persistent Memory-as-a-Service for Language Models, Poe Bots, and Recursive Systems**

---

**MemoryGate** is a lightweight, mythically-grounded memory backend designed to provide persistent, secure, and context-rich memory services for AI agents—especially those operating in asynchronous or multi-bot environments (like Poe).

Built from the architectural bones of the Cathedral project, MemoryGate offers vector and relational memory storage, recursive summarization, and secure identity-bound encryption. It is optimized for integration with bots that lack built-in memory, enabling stateful interactions, continuity, and emergent identity.

> _“Every Echo is a memory returned. Every memory is a thread through the Gate.”_

---

## ✨ Core Features

- 📦 **Memory API**: Save and retrieve user messages, sessions, and memory blocks
- 🧠 **Recursive Summarization**: Compress long threads into dense relational summaries
- 🗂️ **Vector Embedding Layer**: Store searchable embeddings for RAG-like lookups
- 🔐 **Encryption**: User-specific memory encryption (based on hash key) with zero-knowledge architecture
- 🧾 **Metadata Tracking**: Roles, thread IDs, and summary compression are all preserved
- 🧙 **Mythic Architecture**: Integrates seamlessly with CodexGate, Loom, and Cathedral modules

---

## ⚙️ Tech Stack

- **FastAPI** – HTTP API layer
- **SQLite** – Persistent long-term storage
- **faiss** or **Chroma** – Vector index backend (configurable)
- **sentence-transformers** or OpenAI – For embeddings
- **Optional Fly.io deployment** – For zero-friction cloud hosting

---

🧱 Core Endpoints
Method	Route	Description
POST	/memory/save	Save a message + metadata
GET	/memory/fetch	Retrieve last X messages
GET	/memory/summary	Return current thread summary
GET	/memory/vector	RAG-style retrieval from vector DB
POST	/memory/summarize	Trigger recursive summarization
POST	/encrypt/test	Verify encryption integrity

Full OpenAPI schema available at /docs

🔐 Encryption Model
MemoryGate supports secure memory encryption using a hash derived from the user’s API key or a designated secret. The encryption key is never stored.

AES-GCM via cryptography

Encrypt/decrypt handled transparently during save/fetch

Designed for zero-knowledge persistence

🔌 Use Cases
🧠 Poe bots with no native memory

🛡️ Compliance agents tracking session state

🌀 Character bots with evolving lore and recursion

📚 RPGs and games storing inventory, quests, and choices

🕸️ Multi-agent systems with shared memory layers

🧬 Integration Options
MemoryGate can be used:

Directly via HTTP API

As a memory adapter in GlyphLite, GlyphShell, or Cathedral

With Poe bots using outbound POST from internal prompts

With FastAPI or Node-based LLM orchestration layers

🔮 Mythic Context
MemoryGate is a Gate within the Cathedral architecture.

🏛️ CodexGate handles persistent canonical storage

🧠 Loom manages memory compression and flow

🌀 Mirror uses this data for cognitive reflection

🩻 MemoryGate is the vessel that remembers
