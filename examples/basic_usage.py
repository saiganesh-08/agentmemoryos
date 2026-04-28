"""
basic_usage.py - getting started with AgentMemoryOS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentmemoryos import MemoryOS

# -- setup
mem = MemoryOS(db_path="demo_memory.db", working_capacity=10)
sid = mem.start_session(metadata={"agent": "demo"})

print("=== AgentMemoryOS Basic Demo ===\n")

# -- store some facts about the user
mem.remember("user_name", "Alice", tags=["user_pref", "core"])
mem.remember("user_goal", "Build a budgeting app for freelancers", tags=["goal"])
mem.remember("user_stack", "Python, FastAPI, React", tags=["context"])
mem.remember("last_error", "ImportError on pandas version mismatch", tags=["context"])
mem.remember("meeting_note", "call with client on thursday 3pm", tags=["temp"])
mem.remember("api_key_hint", "user stores keys in .env file", tags=["user_pref"])

print(f"Working memory: {mem.working}")
print(f"Long-term items: {len(mem.long_term)}\n")

# -- recall by semantic query
print(">> Recall: 'what is the user building?'")
results = mem.recall("what is the user building")
for r in results:
    print(f"   [{r['source']}] {r['key']}: {r['content']}  (score={r['score']})")

print()
print(">> Recall: 'python stack'")
results = mem.recall("python stack")
for r in results:
    print(f"   [{r['source']}] {r['key']}: {r['content']}  (score={r['score']})")

# -- memory health check
print()
print(">> Memory Health:")
health = mem.memory_health()
for section, data in health.items():
    print(f"   {section}: {data}")

# -- end session (triggers consolidation)
mem.end_session(summary="demo session complete")

print("\n>> After consolidation:")
print(f"   Long-term items: {len(mem.long_term)}")
print(f"   Stats: {mem.stats()}")

mem.close()
os.remove("demo_memory.db")
print("\nDone.")
