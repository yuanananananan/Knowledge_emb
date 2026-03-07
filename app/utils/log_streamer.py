import os
from datetime import datetime

def append_log(job_id, message):
    if not message.strip():
        return
    log_dir = "D:/Project/knowledge_emb/logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{job_id}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message.strip()}\n")