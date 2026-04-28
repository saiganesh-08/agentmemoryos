import argparse
import json
from .memory_os import MemoryOS


def main():
    parser = argparse.ArgumentParser(
        prog="agentmemory",
        description="AgentMemoryOS CLI - inspect and manage agent memory"
    )
    subparsers = parser.add_subparsers(dest="command")

    # health command
    health_p = subparsers.add_parser("health", help="show memory health snapshot")
    health_p.add_argument("--db", default="agentmemory.db")

    # list command
    list_p = subparsers.add_parser("list", help="list long-term memories")
    list_p.add_argument("--db", default="agentmemory.db")
    list_p.add_argument("--top", type=int, default=20)

    # recall command
    recall_p = subparsers.add_parser("recall", help="query memory")
    recall_p.add_argument("query", type=str)
    recall_p.add_argument("--db", default="agentmemory.db")
    recall_p.add_argument("--top", type=int, default=5)

    # forget command
    forget_p = subparsers.add_parser("forget", help="delete a memory by key")
    forget_p.add_argument("key")
    forget_p.add_argument("--db", default="agentmemory.db")

    # retention report
    ret_p = subparsers.add_parser("retention", help="show retention report")
    ret_p.add_argument("--db", default="agentmemory.db")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    mem = MemoryOS(db_path=args.db)

    try:
        if args.command == "health":
            print(json.dumps(mem.memory_health(), indent=2))

        elif args.command == "list":
            items = mem.long_term.top_by_importance(n=args.top)
            for item in items:
                print(f"{item.key:40s}  importance={item.importance:.2f}  strength={item.memory_strength:.2f}")

        elif args.command == "recall":
            results = mem.recall(args.query, top_k=args.top)
            if not results:
                print("No memories found.")
            for r in results:
                print(f"[{r['source']}] {r['key']} (score={r['score']})")
                print(f"  {r['content']}")

        elif args.command == "forget":
            mem.forget(args.key)
            print(f"Forgot: {args.key}")

        elif args.command == "retention":
            report = mem.retention_report()
            for entry in report:
                hrs = entry["hours_until_forgotten"]
                hrs_str = f"{hrs:.0f}h" if hrs >= 0 else "already faded"
                print(f"{entry['key']:40s}  {entry['retention']:.2f}  {hrs_str}")
    finally:
        mem.close()


if __name__ == "__main__":
    main()
