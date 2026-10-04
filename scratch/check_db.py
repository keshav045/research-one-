import sqlite3, json

conn = sqlite3.connect("researchlens.db")
c = conn.cursor()
c.execute("SELECT id, question, status, debug_json FROM research_jobs ORDER BY created_at DESC LIMIT 5")
for row in c.fetchall():
    jid, q, st, dbg = row
    dbg_dict = json.loads(dbg) if dbg else {}
    print(f"JOB: {jid} | {q[:55]} | {st}")
    if "evidence_filter_removals" in dbg_dict:
        print("  evidence_filter_removals:", dbg_dict["evidence_filter_removals"])
    else:
        print("  Keys in debug_json:", list(dbg_dict.keys()))
