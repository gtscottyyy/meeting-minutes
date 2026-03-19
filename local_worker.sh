#!/bin/bash
# local_worker.sh — runs on CT 103, SSHes into Windows and runs --local pipeline
# Cron: 10am + 10pm daily

WINDOWS_HOST="scott@192.168.86.123"
PIPELINE_DIR="Desktop/vimeo"
LOG=/opt/meeting-minutes/logs/local_worker.log
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

# Load secrets from .env
set -a; source /opt/meeting-minutes/.env; set +a

telegram() {
    curl -sf -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" >/dev/null 2>&1 || true
}

echo "=== $(date) ===" >> "$LOG"

# Check if Windows is reachable
if ! ssh $SSH_OPTS "$WINDOWS_HOST" "echo ok" >/dev/null 2>&1; then
    echo "[worker] Windows not reachable — skipping" >> "$LOG"
    telegram "⚠️ *meeting-minutes* — Windows not reachable for local transcription run"
    exit 0
fi

# Check if there's anything to do
STUCK=$(ssh $SSH_OPTS "$WINDOWS_HOST" \
    "cd $PIPELINE_DIR && python -c \"
import sqlite3, os
db = 'data/videos.db' if os.path.exists('data/videos.db') else 'videos.db'
try:
    conn = sqlite3.connect(db)
    n = conn.execute(\\\"SELECT COUNT(*) FROM videos WHERE status='audio_ready'\\\").fetchone()[0]
    print(n)
except: print(0)
\"" 2>/dev/null || echo "0")

if [[ "$STUCK" == "0" ]]; then
    echo "[worker] No audio_ready videos — skipping" >> "$LOG"
    exit 0
fi

echo "[worker] ${STUCK} audio_ready video(s) found — starting local run on Windows" >> "$LOG"
telegram "🖥️ *meeting-minutes* — starting local transcription of ${STUCK} video(s) on Windows"

# Run full pipeline on Windows
ssh $SSH_OPTS "$WINDOWS_HOST" "cd $PIPELINE_DIR && git pull && python vimeo_pipeline.py --local" >> "$LOG" 2>&1
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[worker] Pipeline completed successfully" >> "$LOG"

    # Push from Windows
    ssh $SSH_OPTS "$WINDOWS_HOST" "cd $PIPELINE_DIR && git diff --quiet && git diff --cached --quiet || (git add -A && git commit -m 'auto: local transcription $(date +%Y-%m-%d)' && git push)" >> "$LOG" 2>&1

    telegram "✅ *meeting-minutes* — local transcription complete, report updated"
else
    echo "[worker] Pipeline exited with code ${EXIT_CODE}" >> "$LOG"
    telegram "⚠️ *meeting-minutes* — local pipeline exited with errors (code ${EXIT_CODE}), check logs"
fi

echo "=== done ===" >> "$LOG"
