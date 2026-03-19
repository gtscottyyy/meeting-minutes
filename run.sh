#!/bin/bash
cd /opt/meeting-minutes
export PATH=/usr/local/bin:/usr/bin:/bin

LOG=/opt/meeting-minutes/logs/cron.log

# Load secrets from .env
set -a; source /opt/meeting-minutes/.env; set +a

telegram() {
    curl -sf -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=$1" >/dev/null 2>&1 || true
}

echo "=== $(date) ===" >> "$LOG"

# Pull latest changes from GitHub before running
echo "[pull] Pulling latest from GitHub..." >> "$LOG"
git pull >> "$LOG" 2>&1

# Run pipeline (cloud only)
echo "[pipeline] Starting vimeo_pipeline.py --cloud..." >> "$LOG"
python3 vimeo_pipeline.py --cloud >> "$LOG" 2>&1
PIPELINE_EXIT=$?

# Check for videos stuck at audio_ready (need --local on Windows)
STUCK=$(python3 - << 'EOF'
import sqlite3
conn = sqlite3.connect('data/videos.db')
rows = conn.execute("SELECT title FROM videos WHERE status='audio_ready'").fetchall()
for r in rows: print(r[0])
EOF
)

if [[ -n "$STUCK" ]]; then
    COUNT=$(echo "$STUCK" | wc -l)
    echo "[alert] ${COUNT} video(s) stuck at audio_ready -- need --local:" >> "$LOG"
    echo "$STUCK" >> "$LOG"
    LIST=$(echo "$STUCK" | sed 's/^/• /')
    telegram "🎙 *meeting-minutes* — ${COUNT} video(s) stuck, Windows will pick up at next 10am/10pm run:
${LIST}"
else
    echo "[alert] No videos stuck at audio_ready" >> "$LOG"
fi

# Commit and push if anything changed
if git diff --quiet && git diff --cached --quiet; then
    echo "[git] No changes to commit" >> "$LOG"
else
    git add -A
    git commit -m "auto: update report $(date +%Y-%m-%d)"
    git push >> "$LOG" 2>&1
    echo "[git] Pushed changes" >> "$LOG"
fi

echo "=== done (exit ${PIPELINE_EXIT}) ===" >> "$LOG"
