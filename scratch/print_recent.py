import json

log_file = r"C:\Users\kesha\.gemini\antigravity-ide\brain\a9e648d8-e845-4581-9203-971663ae63f6\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        if s.get("step_index") == 1891:
            print(s.get("content"))
