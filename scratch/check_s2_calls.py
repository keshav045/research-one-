import re
import urllib.parse

log_path = r"C:\Users\kesha\.gemini\antigravity-ide\brain\a9e648d8-e845-4581-9203-971663ae63f6\.system_generated\tasks\task-1755.log"
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# The questions in order
sections = re.split(r"RUNNING BENCHMARK FOR:\s*", text)
for sec in sections[1:]:
    topic_line = sec.split("\n")[0].strip()
    print(f"\n=======================================================")
    print(f"Topic: {topic_line}")
    print(f"=======================================================")
    
    # Extract query requests and statuses
    lines = sec.splitlines()
    query_attempts = {}
    for l in lines:
        m = re.search(r"query=([^&\s]+)&limit=\d+.*?HTTP/1\.1\s+(\d+)", l)
        if m:
            raw_q, status = m.groups()
            dec_q = urllib.parse.unquote_plus(raw_q)
            if dec_q not in query_attempts:
                query_attempts[dec_q] = []
            query_attempts[dec_q].append(status)
            
    for q, attempts in query_attempts.items():
        # Check if query returned papers
        pat = rf"\[S2\] Query '{re.escape(q)}' -> (\d+) papers"
        m_count = re.search(pat, sec)
        count = int(m_count.group(1)) if m_count else 0
        final_status = "200 OK" if "200" in attempts else "429 Rate Limited"
        print(f"- Query: '{q}'")
        print(f"  Attempts: {' -> '.join(['HTTP ' + a for a in attempts])}")
        print(f"  Final Status: {final_status} | Papers Returned: {count}")
