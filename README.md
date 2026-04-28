# AgentMemoryOS

![tests](https://github.com/YOUR_USERNAME/agentmemoryos/actions/workflows/ci.yml/badge.svg)

A memory operating system for AI agents.

Most agent frameworks bolt on a vector database and call it "memory." That's not memory — that's a search index. Real memory has working capacity limits, long-term consolidation, forgetting curves, importance weighting, and episodic context. This project implements all of that.

---

## Why this exists

Building a GPT-based assistant that remembers things across sessions is surprisingly hard. You can stuff everything into a context window (hits limits fast), use a vector DB (no concept of forgetting or salience), or roll your own (which is what this is).

AgentMemoryOS gives your agent:

- **Working memory** — limited capacity buffer with LRU eviction, like actual cognitive working memory
- **Long-term memory** — SQLite-backed persistent store with importance scoring
- **Episodic memory** — session and event tracking, so the agent knows what happened when
- **Forgetting engine** — Ebbinghaus decay curves, memories fade unless reinforced
- **Semantic retrieval** — cosine similarity search, plug in your own embeddings if you want better quality
- **Auto-consolidation** — important things get promoted from working to long-term automatically

---

## Install

```bash
git clone https://github.com/yourusername/agentmemoryos
cd agentmemoryos
pip install -r requirements.txt
```

For better semantic search (optional):
```bash
pip install sentence-transformers
```

---

## Quick start

```python
from agentmemoryos import MemoryOS

mem = MemoryOS(db_path="my_agent.db")
sid = mem.start_session()

mem.remember("user_name", "Alice", tags=["user_pref", "core"])
mem.remember("user_goal", "build a budgeting app", tags=["goal"])
mem.remember("last_error", "pandas version mismatch", tags=["context"])

results = mem.recall("what is the user trying to build")
for r in results:
    print(r["key"], "->", r["content"], f"(score={r['score']})")

mem.end_session()
mem.close()
```

---

## Using your own embeddings

The default bag-of-words search works okay but won't win any benchmarks. Pass in any embedding function:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

mem = MemoryOS(
    embed_fn=lambda text: model.encode(text)
)
```

Works with OpenAI embeddings too:

```python
import openai

def embed(text):
    res = openai.embeddings.create(input=text, model="text-embedding-3-small")
    return res.data[0].embedding

mem = MemoryOS(embed_fn=embed)
```

---

## Memory health

```python
print(mem.memory_health())
# {
#   "working_memory": {"used": 8, "capacity": 20},
#   "long_term_memory": {"total_items": 42, "avg_strength": 0.73, "at_risk": ["old_note_1"]},
#   "episodic": {"total_sessions": 5, "active_session": "a3f2"},
#   ...
# }
```

---

## Forgetting

Memories decay over time using an Ebbinghaus-style curve. More important memories and ones accessed frequently decay slower.

```python
report = mem.retention_report()
# shows each memory's retention score and estimated hours until it fades
```

You can also trigger manual consolidation:

```python
result = mem.consolidate()
print(result["promoted"])  # items moved from working to long-term
print(result["forgotten"])  # items that decayed past threshold
```

---

## CLI

```bash
# health check
agentmemory health --db my_agent.db

# list top memories by importance
agentmemory list --db my_agent.db --top 10

# semantic query
agentmemory recall "what does the user prefer" --db my_agent.db

# retention report
agentmemory retention --db my_agent.db

# delete a memory
agentmemory forget user_name --db my_agent.db
```

---

## Run tests

```bash
pytest tests/ -v
```

---

## Project structure

```
agentmemoryos/
├── agentmemoryos/
│   ├── memory_os.py          # main interface
│   ├── core/
│   │   ├── working_memory.py
│   │   ├── long_term_memory.py
│   │   └── episodic_memory.py
│   ├── engine/
│   │   ├── forgetting.py     # Ebbinghaus decay
│   │   ├── importance.py     # importance scoring
│   │   └── consolidator.py
│   ├── retrieval/
│   │   └── semantic.py       # cosine similarity search
│   ├── storage/
│   │   └── sqlite_store.py
│   └── cli.py
├── examples/
│   ├── basic_usage.py
│   └── chatbot_agent.py
└── tests/
    ├── test_working_memory.py
    └── test_forgetting.py
```

---

## Limitations

- Semantic search quality depends heavily on the embedding function. The default BoW approach is fine for exact keyword matches but misses synonyms and paraphrases.
- SQLite is fine for single-agent use, but if you're running concurrent agents you'll hit locking issues. Swap the storage backend for something like Redis or Postgres.
- The importance scorer is heuristic-based. It works reasonably well but you can override it by passing `importance=` directly to `remember()`.
- No distributed / multi-agent memory sharing yet. That's on the roadmap.

---

## License

MIT
