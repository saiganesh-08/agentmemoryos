"""
chatbot_agent.py

Simulates a multi-turn chatbot that remembers things across the conversation.
Shows how AgentMemoryOS would plug into a real agent loop.

This doesn't call any real LLM — it just simulates the memory layer behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from agentmemoryos import MemoryOS


class SimpleAgent:
    def __init__(self, name="Agent"):
        self.name = name
        self.mem = MemoryOS(
            db_path=f"{name.lower()}_memory.db",
            working_capacity=15,
            auto_consolidate=True,
            consolidate_every=5,
        )
        self.session_id = self.mem.start_session()
        print(f"[{self.name}] Session started: {self.session_id}\n")

    def process(self, user_input: str) -> str:
        # log the user input as an episodic event
        self.mem.episodic.log("user_input", user_input)

        # check if we know anything relevant
        memories = self.mem.recall(user_input, top_k=3)

        context_parts = []
        for m in memories:
            if m["score"] > 0.1:
                context_parts.append(f"{m['key']}: {m['content']}")

        # simulate extracting facts from user input and storing them
        self._extract_and_store(user_input)

        # build a fake response (in real usage, you'd pass context to an LLM here)
        if context_parts:
            response = f"[Using context: {'; '.join(context_parts)}] Got it."
        else:
            response = "Noted, I'll remember that."

        self.mem.episodic.log("agent_response", response)
        return response

    def _extract_and_store(self, text: str):
        # super naive extraction — real impl would use NLP/LLM
        text_lower = text.lower()

        if "my name is" in text_lower:
            name = text.split("my name is", 1)[1].strip().split()[0]
            self.mem.remember(f"user_name", name, tags=["user_pref", "core"])

        if "i prefer" in text_lower or "i like" in text_lower:
            self.mem.remember(f"pref_{int(time.time())}", text, tags=["user_pref"])

        if "remember" in text_lower:
            self.mem.remember(f"note_{int(time.time())}", text, tags=["core"])

        if "goal" in text_lower or "i want to" in text_lower:
            self.mem.remember(f"goal_{int(time.time())}", text, tags=["goal"])

    def shutdown(self):
        print(f"\n[{self.name}] Ending session...")
        self.mem.end_session(summary="user conversation complete")
        health = self.mem.memory_health()
        print(f"[{self.name}] Memory health: {health}")
        self.mem.close()
        if os.path.exists(f"{self.name.lower()}_memory.db"):
            os.remove(f"{self.name.lower()}_memory.db")


def main():
    agent = SimpleAgent("Jarvis")

    # simulate a conversation
    conversation = [
        "Hey, my name is Bob",
        "I prefer Python over JavaScript",
        "I want to build a habit tracker app",
        "Can you remember that I have a meeting on Monday?",
        "What do you know about me?",
        "I like dark mode and minimal UIs",
        "What's my goal again?",
    ]

    for turn, msg in enumerate(conversation):
        print(f"User: {msg}")
        response = agent.process(msg)
        print(f"Agent: {response}")
        print()

    # show retention report
    print("--- Retention Report ---")
    report = agent.mem.retention_report()
    for entry in report[:5]:
        hrs = entry["hours_until_forgotten"]
        hrs_str = f"{hrs:.0f}h" if hrs > 0 else "already decayed"
        print(f"  {entry['key']:30s}  retention={entry['retention']:.2f}  fades in: {hrs_str}")

    agent.shutdown()


if __name__ == "__main__":
    main()
