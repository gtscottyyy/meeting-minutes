#!/bin/bash
cd /opt/meeting-minutes
export PATH=/usr/local/bin:/usr/bin:/bin

LOG=/opt/meeting-minutes/logs/cron.log
UPTIME_KUMA_PUSH_URL=""  # set after creating push monitor in Uptime Kuma

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
    if [[ -n "$UPTIME_KUMA_PUSH_URL" ]]; then
        MSG="${COUNT}+video(s)+need+--local+transcription"
        curl -sf "${UPTIME_KUMA_PUSH_URL}?status=down&msg=${MSG}&ping=" >/dev/null 2>&1 || true
    fi
else
    echo "[alert] No videos stuck at audio_ready" >> "$LOG"
    if [[ -n "$UPTIME_KUMA_PUSH_URL" ]]; then
        curl -sf "${UPTIME_KUMA_PUSH_URL}?status=up&msg=Pipeline+OK&ping=" >/dev/null 2>&1 || true
    fi
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
