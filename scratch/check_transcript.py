import json

log_file = r"C:\Users\kesha\.gemini\antigravity-ide\brain\a9e648d8-e845-4581-9203-971663ae63f6\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("type") == "USER_INPUT":
            print("=" * 60)
            print(obj.get("content"))
