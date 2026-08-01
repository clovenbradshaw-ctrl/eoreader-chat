# EOChat UX Design Document

## Product Vision

**A chat that remembers everything and answers from your actual documents — not hallucinated.**

EOChat is an infinite-memory conversational interface powered by the eoreader5 semantic engine. Users ingest texts (books, notes, URLs), have natural conversations, and receive grounded answers with verifiable citations pointing to real passages in their sources.

---

## User Personas

### The Researcher
- **Goal:** Synthesize insights across multiple long documents
- **Pain:** Can't remember where they read something; loses track of cross-text patterns
- **Need:** "Show me what I read about X across all my sources, with exact quotes"

### The Knowledge Worker
- **Goal:** Build a personal knowledge base that grows over time
- **Pain:** Information scattered across apps; context lost between sessions
- **Need:** "Remember what I told you last week and connect it to what I'm asking now"

### The Writer
- **Goal:** Draft content grounded in source material
- **Pain:** Writing essays/reports requires constant context-switching between sources
- **Need:** "Write a section about X using my sources, cite everything, show me the gaps"

---

## Affordance Catalog

### 1. Natural Conversation

**What it is:** Chat interface where users speak naturally, not in search queries.

**Why users need it:** People think in dialogue, not keywords. A conversational interface reduces cognitive load and matches how humans naturally exchange information.

**How it manifests:**
- Chat UI with message input
- Streaming responses (SSE)
- Conversation history preserved across turns

**Success criteria:**
- User can ask "What did we talk about yesterday?" and get a coherent answer
- Follow-up questions reference prior context without re-explanation
- Response latency < 2s for simple queries, < 5s for complex grounding

**How to test:**
1. Start a new session
2. Say "Hello, my name is Alice"
3. Ask "What's my name?" → should respond "Alice"
4. Ask "What did I just tell you?" → should reference the name
5. Close and reopen → name should persist

---

### 2. Persistent Memory

**What it is:** Every message (user and assistant) is folded into the engine as an observation, searchable forever.

**Why users need it:** Users don't repeat themselves. The system builds on what was said before, creating continuity across sessions.

**How it manifests:**
- `POST /api/discourse/message` — saves messages
- `GET /api/discourse/context` — retrieves conversation history
- `GET /api/discourse/stats` — shows message count, token usage

**Success criteria:**
- Information from 100 turns ago is retrievable
- Memory doesn't degrade over time
- User can reference "that thing I said last week" and the system finds it

**How to test:**
1. Send 50 messages over multiple sessions
2. Ask "What was the third topic I brought up?" → should retrieve it
3. Ask "Summarize our conversation" → should be coherent
4. Check `/api/discourse/stats` → message count matches

---

### 3. Document Ingest

**What it is:** Upload files (txt, pdf, epub) or URLs to make them searchable.

**Why users need it:** Answer from *your* knowledge, not just training data. The system becomes a personal library.

**How it manifests:**
- `POST /api/ingest` — ingest a file path
- `POST /api/chat/tools` with `ingest` tool — LLM can ingest on request
- `GET /extract?url=` (memory-server) — fetch URL content

**Success criteria:**
- User can drop a 1000-page book and search it in < 10s
- Duplicate files are detected (SHA hash) and rejected
- Multiple file formats supported

**How to test:**
1. Ingest `pg84.txt` (Frankenstein, 438KB)
2. Check `/api/sources` → file appears with chunk count
3. Ingest same file again → should return "duplicate" error
4. Ingest a URL → content extracted and stored

---

### 4. Grounded Citations

**What it is:** Every claim in the response is backed by a citation `[1]`, `[2]` pointing to a real passage.

**Why users need it:** Trust. Users can verify every statement by clicking the citation to see the exact source text. No hallucination.

**How it manifests:**
- LLM response includes `[1]`, `[2]` markers
- Citations link to `/api/verbatim/read?span_id=...`
- Citation audit in `/api/chat/tools` response shows which citations were used

**Success criteria:**
- Every factual claim has a citation
- Clicking `[1]` shows the exact passage
- Citations are mechanically verified (not model-generated)

**How to test:**
1. Ask "What does Gregor Samsa turn into?"
2. Response should include `[1]`
3. Click `[1]` → should show Kafka's text about the insect
4. Check citation audit → `used: [1]`, `allGrounded: true`

---

### 5. Cross-Text Reasoning

**What it is:** Search and synthesize across multiple ingested documents simultaneously.

**Why users need it:** See patterns you wouldn't manually compare. "How do Frankenstein's creature and Gregor Samsa both experience isolation?"

**How it manifests:**
- `GET /api/verbatim?q=...` searches all sources in the pool
- LLM receives context from multiple sources
- Response synthesizes across texts

**Success criteria:**
- Query returns passages from all relevant sources
- Response explicitly compares/contrasts sources
- User can ask "Which source says X?" and get a specific answer

**How to test:**
1. Ingest Frankenstein + Metamorphosis
2. Ask "Compare how both protagonists experience isolation"
3. Response should cite both texts
4. Check grounding → citations from both sources present

---

### 6. Verbatim Search

**What it is:** Find exact quotes from ingested texts, byte-offset anchored.

**Why users need it:** "Where did I read that..." — find things again without remembering where. Researchers need exact quotes for citations.

**How it manifests:**
- `GET /api/verbatim?q=<query>&limit=<n>` — search for passages
- `GET /api/verbatim/read?span_id=<id>` — read a specific span
- `GET /api/verbatim/segment?q=<query>` — read surrounding context

**Success criteria:**
- Search returns exact text from source (no paraphrasing)
- Results include byte offsets for precise location
- User can page through long results

**How to test:**
1. Ingest War and Peace
2. Search "Natasha ball dance" → should return exact passage
3. Click result → should show the exact text from the book
4. Verify byte offset matches source file

---

### 7. Drill-Down (FETCH)

**What it is:** LLM can request full content of an attachment or source on demand.

**Why users need it:** First answer isn't always enough. User wants to go deeper without re-asking.

**How it manifests:**
- `fetch_attachment` tool — LLM calls it to read uploaded file
- `verbatim_read` tool — LLM reads a specific span
- `web_fetch` tool — LLM fetches URL content

**Success criteria:**
- LLM can autonomously decide to fetch more context
- User doesn't need to manually trigger drill-down
- Fetched content is integrated into response

**How to test:**
1. Upload a long document
2. Ask a question that requires reading the full doc
3. LLM should call `fetch_attachment` automatically
4. Response should reference content from the full document

---

### 8. Streaming Responses

**What it is:** Responses stream token-by-token via Server-Sent Events (SSE).

**Why users need it:** Know it's working, see it think, don't stare at a blank screen. Reduces perceived latency.

**How it manifests:**
- `POST /api/chat/tools` returns SSE stream
- Events: `grounding`, `llm_call`, `response`, `tool_call`, `done`
- UI renders tokens as they arrive

**Success criteria:**
- First token arrives < 1s
- Stream doesn't block or freeze
- User can see tool calls happening in real-time

**How to test:**
1. Ask a complex question
2. Observe SSE events in network tab
3. Verify `grounding` event arrives first
4. Verify `response` tokens stream smoothly
5. Verify `done` event signals completion

---

### 9. Source Control

**What it is:** Delete, restore, or permanently purge ingested sources.

**Why users need it:** You decide what it knows. Fix mistakes. Manage your knowledge base.

**How it manifests:**
- `DELETE /api/sources/<key>` — soft delete (recycle bin)
- `GET /api/recycle-bin` — list deleted sources
- `POST /api/recycle-bin/restore` — restore from recycle bin
- `DELETE /api/recycle-bin` — permanently purge

**Success criteria:**
- Deleted source no longer appears in search results
- Restored source is searchable again
- Purged source is gone forever

**How to test:**
1. Ingest a file
2. Delete it → should move to recycle bin
3. Search → file should not appear
4. Restore it → file should be searchable again
5. Purge it → file should be gone permanently

---

### 10. Priors (Domain Knowledge)

**What it is:** Pre-loaded domain knowledge that steers retrieval without being cited as evidence.

**Why users need it:** The system should already understand the field, not start from zero. Priors are witness-tier knowledge about the corpus, not evidence from it.

**How it manifests:**
- `GET /api/priors` — list available priors
- `GET /api/priors/read?id=<id>` — read a specific prior
- `GET /api/priors/search?q=<query>` — search priors
- Priors activate automatically on surf (not exposed to LLM as context)

**Success criteria:**
- Priors improve retrieval quality without being cited
- User can browse what priors are loaded
- Priors don't leak into corpus grounding

**How to test:**
1. Check `/api/priors` → should list coref priors
2. Read a prior → should show domain knowledge (e.g., entity mappings)
3. Ask a question → priors should steer retrieval but not appear as citations
4. Verify priors pool is separate from corpus pool

---

### 11. Complex Task Decomposition (Holonic)

**What it is:** Break big asks ("write a 5-page essay") into grounded sub-tasks with mechanical citations.

**Why users need it:** Users want complex outputs, not just Q&A. The system plans, researches, executes, and assembles — all grounded in the engine.

**How it manifests:**
- `POST /api/holonic` — dedicated endpoint for task decomposition
- `holonic_task` tool — LLM can call it during chat
- SSE events: `holonic_plan`, `holonic_research`, `holonic_execute`, `holonic_assemble`, `done`

**Success criteria:**
- Task is decomposed into 4-8 sub-tasks
- Each sub-task is researched via engine
- Output has mechanical citations (not model-generated)
- Missing evidence produces typed gaps, not fake citations

**How to test:**
1. Call `/api/holonic` with "Write an essay about creation in Frankenstein and the Bible"
2. Observe planning phase → should show 4-6 sub-tasks
3. Observe research phase → each sub-task searches engine
4. Observe execution phase → content generated with citations
5. Final output should have `[1]`, `[2]` citations and a references section

---

### 12. Adaptive Model Routing

**What it is:** Simple queries use tiny models; complex queries use bigger models.

**Why users need it:** Don't wait for a big model to say "hello"; don't get a dumb answer to a hard question. Balance speed and quality.

**How it manifests:**
- `model-router.js` — learned routing based on measured outcomes
- `selectModel()` — heuristic fallback based on message complexity
- Models: `phi4-mini:latest` (tiny), `qwen2.5-coder:7b` (medium)

**Success criteria:**
- "Hello" routes to tiny model (< 1s response)
- Code question routes to medium model
- Routing improves over time (learned from outcomes)

**How to test:**
1. Send "Hi" → should use tiny model
2. Send "Refactor this function" → should use medium model
3. Check `/v1/router` → should show routing weights
4. Verify response quality matches complexity

---

### 13. Web Search & Fetch

**What it is:** Real-time web search and URL fetching during chat.

**Why users need it:** Not everything is in your corpus. Sometimes you need current information.

**How it manifests:**
- `web_search` tool — Brave Search API
- `web_fetch` tool — fetch URL content
- LLM can call these autonomously

**Success criteria:**
- LLM can search the web when local knowledge is insufficient
- Fetched content is integrated into response
- User can see what was searched and fetched

**How to test:**
1. Ask "What's the weather today?"
2. LLM should call `web_search`
3. Response should include current information
4. Check tool calls → should show `web_search` and/or `web_fetch`

---

### 14. Codebase Navigation

**What it is:** Tools for exploring and understanding the eoreader5 codebase itself.

**Why users need it:** Developers need to understand the engine they're working with. Self-documenting system.

**How it manifests:**
- `codebase_structure` — show directory tree
- `codebase_find` — find definitions by name
- `codebase_lookup` — get module details
- `codebase_search` — full-text search
- `codebase_related` — show dependencies
- `codebase_entities` — list conceptual entities
- `codebase_api` — show API surface
- `codebase_summary` — overall stats

**Success criteria:**
- Developer can ask "Where is search implemented?" and get file paths
- Developer can ask "What does this module do?" and get a summary
- Tools return accurate, up-to-date information

**How to test:**
1. Call `codebase_structure` → should show repo tree
2. Call `codebase_find` with "search" → should find search modules
3. Call `codebase_lookup` with a path → should show imports/exports
4. Call `codebase_entities` → should list cube, presence, fold, etc.

---

### 15. Terrain Analysis

**What it is:** Structural analysis of ingested files showing which of 9 terrains are detected.

**Why users need it:** Understand the structural nature of a text — is it entity-focused, field-focused, paradigm-focused? Helps users understand what they've ingested.

**How it manifests:**
- `terrain_report` tool — shows terrain analysis for a file
- 9 terrains: Void, Entity, Kind, Field, Link, Network, Atmosphere, Lens, Paradigm
- Born-gate signal check

**Success criteria:**
- Report accurately identifies structural patterns
- Analysis is fully mechanical (no model call)
- User can understand what the terrain means

**How to test:**
1. Ingest a file
2. Call `terrain_report` with the file path
3. Report should show detected terrains
4. Verify Born-gate signal check passes

---

## Testing Matrix

| Affordance | Happy Path | Edge Case | Failure Mode |
|---|---|---|---|
| Natural Conversation | Follow-up question | 100-turn history | Context loss |
| Persistent Memory | Recall from last week | 1000 messages | Memory degradation |
| Document Ingest | 1000-page book | Duplicate file | Ingest failure |
| Grounded Citations | Every claim cited | No evidence available | Fake citations |
| Cross-Text Reasoning | Compare 2 texts | 10 texts | Context overflow |
| Verbatim Search | Exact quote | No match | Wrong passage |
| Drill-Down | Fetch full doc | Large file | Timeout |
| Streaming | Smooth token flow | Network interruption | Frozen UI |
| Source Control | Delete/restore | Purge | Data loss |
| Priors | Activate on surf | Conflicting priors | Citation leak |
| Holonic Task | 5-page essay | No evidence | Fake citations |
| Model Routing | Simple/complex | Ambiguous query | Wrong model |
| Web Search | Current info | Rate limit | Stale results |
| Codebase Nav | Find module | Renamed file | Stale index |
| Terrain Analysis | Clear structure | Mixed terrain | Wrong classification |

---

## Success Metrics

### User Experience
- **Time to first token:** < 1s for simple queries
- **Response latency:** < 5s for grounded answers
- **Citation accuracy:** 100% of factual claims cited
- **Memory retention:** 100% of messages retrievable after 30 days

### System Performance
- **Ingest speed:** > 100KB/s for large files
- **Search latency:** < 3s for verbatim search
- **Concurrent users:** Support 10+ simultaneous sessions
- **Memory usage:** < 2GB for 100 ingested books

### Trust & Transparency
- **Citation verifiability:** 100% of citations link to exact source
- **Grounding audit:** Every response shows which citations were used
- **Gap reporting:** Missing evidence produces typed gaps, not fake answers
- **Priors transparency:** User can see what priors are loaded

---

## Out of Scope (v1)

- Multi-modal input (images, audio)
- Collaborative editing
- Export to specific formats (PDF, DOCX)
- Mobile apps
- Offline mode
- Multi-language support (English only)

---

## Appendix: API Endpoints

### Chat & Conversation
- `POST /api/chat/tools` — main chat endpoint with tool-calling
- `POST /api/holonic` — dedicated holonic task endpoint
- `POST /api/discourse/message` — save message to discourse
- `GET /api/discourse/context` — get conversation history
- `GET /api/discourse/stats` — discourse statistics

### Document Management
- `POST /api/ingest` — ingest a file
- `GET /api/sources` — list ingested sources
- `DELETE /api/sources/<key>` — delete source (recycle bin)
- `GET /api/recycle-bin` — list deleted sources
- `POST /api/recycle-bin/restore` — restore source
- `DELETE /api/recycle-bin` — purge recycle bin

### Search & Retrieval
- `GET /api/verbatim?q=<query>` — verbatim search
- `GET /api/verbatim/read?span_id=<id>` — read specific span
- `GET /api/verbatim/segment?q=<query>` — read surrounding context
- `GET /api/verbatim/context?span_id=<id>` — get span context
- `GET /api/fold?source=<ref>` — fold projection of a source
- `GET /api/source/text?source=<ref>` — read source text by byte range

### Priors
- `GET /api/priors` — list priors
- `GET /api/priors/read?id=<id>` — read a prior
- `GET /api/priors/search?q=<query>` — search priors

### System
- `GET /health` — health check
- `GET /stats` — engine statistics
- `GET /v1/models` — list available models
- `GET /v1/router` — model routing weights
- `POST /mcp/connect` — connect MCP server
- `GET /mcp/servers` — list MCP servers

### Memory Server (Python)
- `GET /extract?url=<url>` — extract text from URL
- `GET /memory/list` — list memory files
- `GET /memory/<filename>` — serve memory file
- `POST /memory/save` — save memory file
