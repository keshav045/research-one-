import json

log_file = r"C:\Users\kesha\.gemini\antigravity-ide\brain\a9e648d8-e845-4581-9203-971663ae63f6\.system_generated\logs\transcript.jsonl"
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        c = str(s.get("content", ""))
        if "1406.2661" in c or "Generative Adversarial Nets" in c or "Goodfellow" in c:
            print(f"Step {s.get('step_index')}: {c[:250]}")
