"""
Goshen Township Vimeo Pipeline
Discovers, downloads, transcribes, and summarizes public meeting videos.

Requirements:
    pip install yt-dlp faster-whisper anthropic

Usage:
    python vimeo_pipeline.py                    # process all unfinished videos
    python vimeo_pipeline.py --limit 5          # process up to 5 unfinished videos
    python vimeo_pipeline.py --discover         # discover new videos only, no processing
    python vimeo_pipeline.py --report           # regenerate HTML report only
    python vimeo_pipeline.py --video-id 123456  # process one specific video ID
    python vimeo_pipeline.py --cloud            # use OpenAI Whisper API instead of local model
    python vimeo_pipeline.py --cloud --limit 3  # cloud transcription, up to 3 videos
"""

import argparse
import json
import os
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from groq import Groq
from faster_whisper import WhisperModel
import yaml

CLOUD_WHISPER_SIZE_LIMIT = 25 * 1024 * 1024  # 25 MB API limit (Groq)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("config.yaml", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

_twp = _cfg["township"]
VIMEO_URL       = _twp["vimeo_url"]
TOWNSHIP_NAME   = _twp["name"]
TOWNSHIP_COUNTY = _twp["county"]
TOWNSHIP_STATE  = _twp["state"]
TOWNSHIP_ADDRESS = _twp["address"]

REPORT_TITLE    = _cfg["report"]["title"]
REPORT_SUBTITLE = _cfg["report"]["subtitle"]

WHISPER_MODEL_SIZE   = "large-v3-turbo"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# Build Whisper initial prompt from config
def _build_whisper_prompt(cfg: dict) -> str:
    twp = cfg["township"]
    officials = ", ".join(cfg.get("officials", []))
    roads = ", ".join(cfg["local_vocabulary"].get("roads", []))
    places = ", ".join(cfg["local_vocabulary"].get("places", []))
    procedural = ". ".join(p.capitalize() for p in cfg["local_vocabulary"].get("procedural", []))
    return (
        f"{twp['name']} meeting, {twp['county']}, {twp['state']}. "
        f"Officials: {officials}. "
        f"{twp['address']}. "
        f"Roads: {roads}. "
        f"{places}. "
        f"{procedural}."
    )

WHISPER_INITIAL_PROMPT = _build_whisper_prompt(_cfg)

CLAUDE_MODEL = "claude-opus-4-6"

FFMPEG_DIR      = Path(os.environ["FFMPEG_PATH"]) if os.environ.get("FFMPEG_PATH") else None

DATA_DIR        = Path("data")
AUDIO_DIR       = DATA_DIR / "audio"
TRANSCRIPT_DIR  = DATA_DIR / "transcripts"
SUMMARY_DIR     = DATA_DIR / "summaries"
DB_PATH         = DATA_DIR / "videos.db"
REPORT_PATH     = Path("index.html")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db() -> sqlite3.Connection:
    for d in [DATA_DIR, AUDIO_DIR, TRANSCRIPT_DIR, SUMMARY_DIR]:
        d.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id        TEXT PRIMARY KEY,
            title           TEXT,
            upload_date     TEXT,
            duration_sec    INTEGER,
            vimeo_url       TEXT,
            status          TEXT DEFAULT 'discovered',
            audio_path      TEXT,
            transcript_path TEXT,
            summary_path    TEXT,
            discovered_at   TEXT,
            updated_at      TEXT
        )
    """)
    conn.commit()
    return conn


def upsert_video(conn: sqlite3.Connection, video: dict):
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT INTO videos (video_id, title, upload_date, duration_sec, vimeo_url,
                            status, discovered_at, updated_at)
        VALUES (:video_id, :title, :upload_date, :duration_sec, :vimeo_url,
                'discovered', :now, :now)
        ON CONFLICT(video_id) DO UPDATE SET
            title        = excluded.title,
            upload_date  = excluded.upload_date,
            duration_sec = excluded.duration_sec,
            updated_at   = :now
    """, {**video, "now": now})
    conn.commit()


def set_status(conn: sqlite3.Connection, video_id: str, status: str, **extra):
    now    = datetime.utcnow().isoformat()
    fields = (", " + ", ".join(f"{k} = :{k}" for k in extra)) if extra else ""
    conn.execute(
        f"UPDATE videos SET status = :status, updated_at = :now{fields} WHERE video_id = :video_id",
        {"status": status, "now": now, "video_id": video_id, **extra}
    )
    conn.commit()


def get_unfinished(conn: sqlite3.Connection, limit: int = 0) -> list:
    q = "SELECT * FROM videos WHERE status != 'summarized' ORDER BY upload_date DESC"
    if limit:
        q += f" LIMIT {limit}"
    return conn.execute(q).fetchall()


def get_video(conn: sqlite3.Connection, video_id: str):
    return conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()


def get_all_summarized(conn: sqlite3.Connection) -> list:
    return conn.execute(
        "SELECT * FROM videos WHERE status = 'summarized' ORDER BY upload_date DESC"
    ).fetchall()

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_videos(conn: sqlite3.Connection) -> int:
    print(f"[discover] Fetching video list from {VIMEO_URL} ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--flat-playlist",
            "--print", "%(id)s\t%(title)s\t%(upload_date)s\t%(duration)s\t%(webpage_url)s",
            VIMEO_URL,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[discover] ERROR: {result.stderr.strip()}")
        return 0

    new_count = 0
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        video_id, title, upload_date, duration, url = parts[:5]
        try:
            duration_sec = int(float(duration)) if duration and duration != "NA" else 0
        except ValueError:
            duration_sec = 0

        if not get_video(conn, video_id):
            upsert_video(conn, {
                "video_id":     video_id,
                "title":        title,
                "upload_date":  upload_date or "",
                "duration_sec": duration_sec,
                "vimeo_url":    url,
            })
            new_count += 1

    total = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"[discover] {new_count} new video(s) found. {total} total in manifest.")
    return new_count

# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def safe_name(title: str) -> str:
    """Convert a video title to a safe filename stem, e.g. 'BOT 3-10-26'."""
    # Remove characters that are invalid in Windows filenames
    safe = re.sub(r'[\\/:*?"<>|]', "", title)
    return safe.strip()


def rename_files(conn: sqlite3.Connection) -> None:
    """One-time migration: rename video_id-based files to title-based names."""
    cur = conn.execute("SELECT * FROM videos WHERE status != 'discovered'")
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        vid   = row["video_id"]
        title = row["title"]
        stem  = safe_name(title)

        new_audio      = AUDIO_DIR      / f"{stem}.mp3"
        new_transcript = TRANSCRIPT_DIR / f"{stem}.txt"
        new_summary    = SUMMARY_DIR    / f"{stem}.json"

        old_audio      = Path(row["audio_path"])      if row["audio_path"]      else AUDIO_DIR      / f"{vid}.mp3"
        old_transcript = Path(row["transcript_path"]) if row["transcript_path"] else TRANSCRIPT_DIR / f"{vid}.txt"
        old_summary    = Path(row["summary_path"])    if row["summary_path"]    else SUMMARY_DIR    / f"{vid}.json"

        def mv(src, dst):
            if src.exists() and src != dst:
                dst.parent.mkdir(exist_ok=True)
                src.rename(dst)
                print(f"  {src.name} → {dst.name}")

        mv(old_audio, new_audio)
        mv(old_transcript, new_transcript)
        mv(old_summary, new_summary)

        conn.execute(
            """UPDATE videos SET
                audio_path      = CASE WHEN audio_path      IS NOT NULL THEN ? ELSE NULL END,
                transcript_path = CASE WHEN transcript_path IS NOT NULL THEN ? ELSE NULL END,
                summary_path    = CASE WHEN summary_path    IS NOT NULL THEN ? ELSE NULL END
               WHERE video_id = ?""",
            (str(new_audio), str(new_transcript), str(new_summary), vid),
        )
        updated += 1

    conn.commit()
    print(f"[rename] Migrated {updated} video(s).")


def recompress_audio(conn: sqlite3.Connection) -> None:
    """Compress oversized audio files that are missing transcripts."""
    rows = conn.execute(
        "SELECT * FROM videos WHERE status = 'audio_ready' AND audio_path IS NOT NULL"
    ).fetchall()

    ffmpeg = str(FFMPEG_DIR / "ffmpeg.exe") if FFMPEG_DIR else "ffmpeg"
    processed = 0
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue
        file_size = audio_path.stat().st_size
        if file_size <= CLOUD_WHISPER_SIZE_LIMIT:
            print(f"[recompress] {audio_path.name} is {file_size/1024/1024:.1f} MB — skipping.")
            continue

        print(f"[recompress] Compressing {audio_path.name} ({file_size/1024/1024:.1f} MB)...")
        tmp = audio_path.with_stem(audio_path.stem + "_tmp")
        result = subprocess.run(
            [ffmpeg, "-i", str(audio_path), "-ac", "1", "-b:a", "32k", str(tmp), "-y"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and tmp.exists():
            new_size = tmp.stat().st_size
            audio_path.unlink()
            tmp.rename(audio_path)
            print(f"[recompress] {file_size/1024/1024:.1f} MB → {new_size/1024/1024:.1f} MB")
            processed += 1
        else:
            print(f"[recompress] Failed: {result.stderr.strip()[:200]}")
            if tmp.exists():
                tmp.unlink()

    print(f"[recompress] Done. {processed} file(s) compressed.")


def fix_json_durations(conn: sqlite3.Connection) -> None:
    """Patch duration_sec into existing summary JSON files using DB values."""
    rows = conn.execute(
        "SELECT video_id, summary_path, duration_sec FROM videos WHERE summary_path IS NOT NULL AND duration_sec > 0"
    ).fetchall()
    updated = 0
    for row in rows:
        p = Path(row["summary_path"])
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("duration_sec", 0) != row["duration_sec"]:
            data["duration_sec"] = row["duration_sec"]
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"  {p.name}: {row['duration_sec']}s")
            updated += 1
    print(f"[fix-json-durations] Patched {updated} file(s).")


def fix_durations(conn: sqlite3.Connection) -> None:
    """Backfill duration_sec for all videos that have an audio file."""
    rows = conn.execute(
        "SELECT video_id, audio_path FROM videos WHERE audio_path IS NOT NULL"
    ).fetchall()
    ffprobe = str(FFMPEG_DIR / "ffprobe.exe") if FFMPEG_DIR else "ffprobe"
    updated = 0
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
            capture_output=True, text=True,
        )
        try:
            duration_sec = int(float(probe.stdout.strip()))
            conn.execute("UPDATE videos SET duration_sec = ? WHERE video_id = ?",
                         (duration_sec, row["video_id"]))
            print(f"  {audio_path.name}: {duration_sec}s")
            updated += 1
        except Exception:
            print(f"  {audio_path.name}: could not read duration")
    conn.commit()
    print(f"[fix-durations] Updated {updated} video(s).")


# ---------------------------------------------------------------------------
# Audio download
# ---------------------------------------------------------------------------

def download_audio(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    video_id = row["video_id"]
    out_path = AUDIO_DIR / f"{safe_name(row['title'])}.mp3"

    if out_path.exists():
        set_status(conn, video_id, "audio_ready", audio_path=str(out_path))
        return True

    print(f"[audio] Downloading {video_id} — {row['title']}")
    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "5",   # 0=best 9=worst; 5 is fine for speech
            *(["--ffmpeg-location", str(FFMPEG_DIR)] if FFMPEG_DIR else []),
            "--output", str(AUDIO_DIR / f"{safe_name(row['title'])}.%(ext)s"),
            "--no-playlist",
            "--quiet",
            row["vimeo_url"],
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not out_path.exists():
        print(f"[audio] ERROR: {result.stderr.strip()[:200]}")
        return False

    # Compress to mono 32kbps if over cloud size limit
    file_size = out_path.stat().st_size
    if file_size > CLOUD_WHISPER_SIZE_LIMIT:
        compressed_path = out_path.with_stem(out_path.stem + "_compressed")
        ffmpeg = str(FFMPEG_DIR / "ffmpeg.exe") if FFMPEG_DIR else "ffmpeg"
        comp = subprocess.run(
            [ffmpeg, "-i", str(out_path), "-ac", "1", "-b:a", "32k", str(compressed_path), "-y"],
            capture_output=True, text=True,
        )
        if comp.returncode == 0 and compressed_path.exists():
            new_size = compressed_path.stat().st_size
            print(f"[audio] Compressed {file_size/1024/1024:.1f} MB → {new_size/1024/1024:.1f} MB (mono 32kbps)")
            out_path.unlink()
            compressed_path.rename(out_path)
        else:
            print(f"[audio] Compression failed, keeping original. {comp.stderr.strip()[:200]}")

    # Get real duration via ffprobe
    ffprobe = str(FFMPEG_DIR / "ffprobe.exe") if FFMPEG_DIR else "ffprobe"
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
        capture_output=True, text=True,
    )
    duration_sec = 0
    try:
        duration_sec = int(float(probe.stdout.strip()))
    except Exception:
        pass

    conn.execute("UPDATE videos SET duration_sec = ? WHERE video_id = ?", (duration_sec, video_id))
    conn.commit()

    set_status(conn, video_id, "audio_ready", audio_path=str(out_path))
    print(f"[audio] Saved {out_path}")
    return True

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

_whisper_model: WhisperModel | None = None

def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        print(f"[transcribe] Loading Whisper {WHISPER_MODEL_SIZE} ({WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}) ...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


def _delete_audio(row: sqlite3.Row) -> None:
    a = row["audio_path"]
    if a:
        p = Path(a)
        if p.exists():
            p.unlink()
            print(f"[audio] Deleted {p.name}")


def transcribe(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    video_id        = row["video_id"]
    transcript_path = TRANSCRIPT_DIR / f"{safe_name(row['title'])}.txt"

    if transcript_path.exists():
        set_status(conn, video_id, "transcribed", transcript_path=str(transcript_path))
        return True

    audio_path = row["audio_path"]
    if not audio_path or not Path(audio_path).exists():
        print(f"[transcribe] Audio not found for {video_id}, skipping.")
        return False

    print(f"[transcribe] Transcribing {video_id} — {row['title']}")
    model = get_whisper_model()

    segments, info = model.transcribe(
        audio_path,
        language="en",
        initial_prompt=WHISPER_INITIAL_PROMPT,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    transcript = "\n".join(lines)
    transcript_path.write_text(transcript, encoding="utf-8")

    set_status(conn, video_id, "transcribed", transcript_path=str(transcript_path))
    print(f"[transcribe] Saved {transcript_path} ({len(transcript):,} chars, lang={info.language})")
    _delete_audio(row)
    return True

def transcribe_cloud(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    video_id        = row["video_id"]
    transcript_path = TRANSCRIPT_DIR / f"{safe_name(row['title'])}.txt"

    if transcript_path.exists():
        set_status(conn, video_id, "transcribed", transcript_path=str(transcript_path))
        return True

    audio_path = row["audio_path"]
    if not audio_path or not Path(audio_path).exists():
        print(f"[transcribe] Audio not found for {video_id}, skipping.")
        return False

    file_size = Path(audio_path).stat().st_size
    if file_size > CLOUD_WHISPER_SIZE_LIMIT:
        print(f"[transcribe] {row['title']} is {file_size / 1024 / 1024:.1f} MB — over 25 MB cloud limit.")
        return False

    print(f"[transcribe] Cloud transcribing {row['title']} ({file_size / 1024 / 1024:.1f} MB)")

    # Try Groq first (fast, free tier)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
            with open(audio_path, "rb") as f:
                result = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    language="en",
                    prompt=WHISPER_INITIAL_PROMPT,
                )
            transcript = result.text.strip()
            transcript_path.write_text(transcript, encoding="utf-8")
            set_status(conn, video_id, "transcribed", transcript_path=str(transcript_path))
            print(f"[transcribe] Saved {transcript_path} ({len(transcript):,} chars) [groq]")
            _delete_audio(row)
            return "groq"
        except Exception as e:
            msg = str(e)
            if "rate_limit_exceeded" in msg or "429" in msg:
                print(f"[transcribe] Groq rate limited — falling back to OpenAI...")
            else:
                print(f"[transcribe] Groq unavailable — falling back to OpenAI...")

    # Fallback to OpenAI
    import openai
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            oa_client = openai.OpenAI(api_key=openai_key)
            with open(audio_path, "rb") as f:
                result = oa_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                    prompt=WHISPER_INITIAL_PROMPT,
                )
            transcript = result.text.strip()
            transcript_path.write_text(transcript, encoding="utf-8")
            set_status(conn, video_id, "transcribed", transcript_path=str(transcript_path))
            print(f"[transcribe] Saved {transcript_path} ({len(transcript):,} chars) [openai]")
            _delete_audio(row)
            return "openai"
        except Exception as e:
            print(f"[transcribe] OpenAI failed ({e}).")

    return None


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = f"""\
You are analyzing a {TOWNSHIP_NAME} ({TOWNSHIP_COUNTY}, {TOWNSHIP_STATE}) \
public meeting transcript.

Extract and return a JSON object with these exact fields:
{{
  "meeting_date":    "YYYY-MM-DD or empty string if unknown",
  "meeting_type":    "Regular Meeting | Special Meeting | Work Session | Other",
  "summary":         "2-4 sentence plain English summary of the meeting",
  "decisions":       ["each motion that passed or failed as a short string"],
  "action_items":    ["tasks assigned to specific people to follow up on"],
  "key_amounts":     ["dollar amounts or financial figures with context"],
  "notable_topics":  ["main agenda topics covered"],
  "attendees":       ["names of trustees, officials, or public speakers mentioned"],
  "sentiment":       "routine | contentious | urgent | ceremonial",
  "notes":           "anything unusual not captured above, or empty string"
}}

Return ONLY valid JSON. No markdown fences, no explanation.\
"""


def summarize(conn: sqlite3.Connection, row: sqlite3.Row, client: anthropic.Anthropic) -> bool:
    video_id     = row["video_id"]
    summary_path = SUMMARY_DIR / f"{safe_name(row['title'])}.json"

    if summary_path.exists():
        set_status(conn, video_id, "summarized", summary_path=str(summary_path))
        return True

    transcript_path = row["transcript_path"]
    if not transcript_path or not Path(transcript_path).exists():
        print(f"[summarize] Transcript not found for {video_id}, skipping.")
        return False

    transcript = Path(transcript_path).read_text(encoding="utf-8")

    # Truncate very long transcripts — Claude handles large contexts well but
    # government meetings rarely need more than ~80k chars.
    max_chars = 80_000
    if len(transcript) > max_chars:
        print(f"[summarize] Truncating transcript from {len(transcript):,} to {max_chars:,} chars")
        transcript = transcript[:max_chars] + "\n\n[...transcript truncated...]"

    print(f"[summarize] Summarizing {video_id} — {row['title']}")

    try:
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": (
                    f"{SUMMARY_PROMPT}\n\n"
                    f"---\n"
                    f"Video title: {row['title']}\n"
                    f"Upload date: {row['upload_date']}\n\n"
                    f"Transcript:\n{transcript}"
                )
            }]
        )
    except anthropic.BadRequestError as e:
        print(f"[summarize] Anthropic API error for {video_id}: {e}")
        return False
    except anthropic.APIStatusError as e:
        print(f"[summarize] Anthropic API error for {video_id}: {e}")
        return False

    raw = message.content[0].text.strip()

    # Strip markdown fences if Claude adds them despite instructions
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"```$", "", raw.strip())

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            print(f"[summarize] ERROR: Could not parse JSON for {video_id}")
            print(f"  Raw (first 300 chars): {raw[:300]}")
            return False

    # Attach source metadata
    data.update({
        "video_id":    video_id,
        "title":       row["title"],
        "upload_date": row["upload_date"],
        "duration_sec": row["duration_sec"],
        "vimeo_url":   row["vimeo_url"],
        "audio_path":  row["audio_path"],
    })

    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    set_status(conn, video_id, "summarized", summary_path=str(summary_path))
    print(f"[summarize] Saved {summary_path}")
    return True

# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: int) -> str:
    if not seconds:
        return "?"
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m}m" if h else f"{m}m"


def _fmt_date(date_str: str) -> str:
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%b %d, %Y")
    except Exception:
        return date_str or "Unknown"


def _sentiment_badge(sentiment: str) -> str:
    colors = {
        "routine":     "#4a9eff",
        "contentious": "#ff6b6b",
        "urgent":      "#ffa500",
        "ceremonial":  "#a78bfa",
    }
    color = colors.get(sentiment, "#888")
    return f'<span class="badge" style="background:{color}">{sentiment}</span>'


def build_report(conn: sqlite3.Connection):
    rows = get_all_summarized(conn)
    if not rows:
        print("[report] No summarized videos yet.")
        return

    summaries = []
    for row in rows:
        if row["summary_path"] and Path(row["summary_path"]).exists():
            data = json.loads(Path(row["summary_path"]).read_text(encoding="utf-8"))
            data["transcript_path"] = row["transcript_path"]
            data["audio_path"]      = row["audio_path"]
            data["duration_sec"]    = row["duration_sec"]
            t_path = Path(row["transcript_path"]) if row["transcript_path"] else None
            if t_path and t_path.exists():
                data["transcript_url"] = "data/transcripts/" + quote(t_path.name)
            summaries.append(data)

    total_meetings  = len(summaries)
    total_decisions = sum(len(s.get("decisions", []))   for s in summaries)
    total_actions   = sum(len(s.get("action_items", [])) for s in summaries)
    contentious     = sum(1 for s in summaries if s.get("sentiment") == "contentious")

    cards_html = ""
    for s in summaries:
        decisions_li = "".join(f"<li>{d}</li>" for d in s.get("decisions", []))
        actions_li   = "".join(f"<li>{a}</li>" for a in s.get("action_items", []))
        amounts_li   = "".join(f"<li>{a}</li>" for a in s.get("key_amounts", []))
        topics_tags  = "".join(f'<span class="topic-tag">{t}</span>' for t in s.get("notable_topics", []))

        attendees_block = ""
        attendees = s.get("attendees", [])
        if attendees:
            names = ", ".join(attendees)
            attendees_block = f"""
            <div class="transcript-toggle">
                <button onclick="toggleTranscript('{s['video_id']}-att')" data-open="&#9650; Hide Attendees" data-closed="&#9660; Attendees ({len(attendees)})">&#9660; Attendees ({len(attendees)})</button>
                <div id="tr-{s['video_id']}-att" class="transcript-body" style="display:none">
                    <div style="margin:8px 0;display:flex;flex-wrap:wrap;gap:6px">{"".join(f'<span class="topic-tag">{a}</span>' for a in attendees)}</div>
                </div>
            </div>"""

        transcript_block = ""
        if s.get("transcript_url"):
            transcript_block = f"""
            <div class="transcript-toggle">
                <button onclick="toggleTranscript('{s['video_id']}', '{s['transcript_url']}')" data-open="&#9650; Hide Transcript" data-closed="&#9660; Show Transcript">&#9660; Show Transcript</button>
                <div id="tr-{s['video_id']}" class="transcript-body" style="display:none">
                    <pre id="tr-text-{s['video_id']}">Loading...</pre>
                </div>
            </div>"""

        cards_html += f"""
        <div class="card" data-sentiment="{s.get('sentiment','')}" data-date="{s.get('meeting_date') or s.get('upload_date','')}">
            <div class="card-header">
                <div>
                    <span class="card-date">{_fmt_date(s.get('meeting_date') or s.get('upload_date',''))}</span>
                    <span class="card-type">{s.get('meeting_type','Meeting')}</span>
                    {_sentiment_badge(s.get('sentiment','routine'))}
                </div>
                <div style="text-align:right;display:flex;align-items:center;gap:8px">
                    <span class="card-duration">{_fmt_duration(s.get('duration_sec', 0))}</span>
                    <a href="{s.get('vimeo_url','#')}" target="_blank" style="font-size:0.8rem;padding:3px 10px;border:1px solid var(--accent);border-radius:4px;color:var(--accent);text-decoration:none;white-space:nowrap">&#9654; Vimeo</a>
                </div>
            </div>
            <h3 class="card-title">
                <a href="{s.get('vimeo_url','#')}" target="_blank">{s.get('title','')}</a>
            </h3>
            <p class="card-summary">{s.get('summary','')}</p>
            <div class="topics">{topics_tags}</div>
            <div class="card-cols">
                <div class="col">
                    <h4>Decisions ({len(s.get('decisions',[]))})</h4>
                    <ul>{decisions_li or '<li class="empty">None recorded</li>'}</ul>
                </div>
                <div class="col">
                    <h4>Action Items ({len(s.get('action_items',[]))})</h4>
                    <ul>{actions_li or '<li class="empty">None recorded</li>'}</ul>
                </div>
                <div class="col">
                    <h4>Financials</h4>
                    <ul>{amounts_li or '<li class="empty">None recorded</li>'}</ul>
                </div>
            </div>
            {attendees_block}
            {transcript_block}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{REPORT_TITLE}</title>
<style>
  :root {{
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --surface2: #22263a;
    --border:   #2e3250;
    --text:     #e2e8f0;
    --muted:    #8892aa;
    --accent:   #4a9eff;
    --link:     #7eb8ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.6; }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 24px 32px; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header p {{ color: var(--muted); font-size: 0.875rem; margin-top: 4px; }}

  .stats {{ display: flex; gap: 16px; padding: 24px 32px; flex-wrap: wrap; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 24px; min-width: 130px; }}
  .stat-value {{ font-size: 2rem; font-weight: 700; color: var(--accent); }}
  .stat-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }}

  .controls {{ padding: 0 32px 20px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
  .controls input {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); padding: 8px 14px; font-size: 0.875rem; width: 280px;
  }}
  .controls input:focus {{ outline: none; border-color: var(--accent); }}
  .filter-btn {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    color: var(--muted); padding: 8px 14px; font-size: 0.8rem; cursor: pointer;
  }}
  .filter-btn.active, .filter-btn:hover {{ border-color: var(--accent); color: var(--text); }}

  .cards {{ padding: 0 32px 48px; display: flex; flex-direction: column; gap: 16px; max-width: 1200px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; }}
  .card[data-sentiment="contentious"] {{ border-left: 3px solid #ff6b6b; }}
  .card[data-sentiment="urgent"]      {{ border-left: 3px solid #ffa500; }}
  .card.hidden {{ display: none; }}

  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }}
  .card-date {{ font-weight: 600; font-size: 0.95rem; }}
  .card-type {{ color: var(--muted); font-size: 0.82rem; margin-left: 10px; }}
  .card-duration {{ color: var(--muted); font-size: 0.82rem; }}
  .badge {{ font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em; margin-left: 8px; color: #fff; }}

  .card-title {{ font-size: 1rem; font-weight: 600; margin-bottom: 8px; }}
  .card-summary {{ color: #c5cde0; font-size: 0.9rem; margin-bottom: 12px; }}

  .topics {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }}
  .topic-tag {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; font-size: 0.75rem; padding: 2px 8px; color: var(--muted); }}

  .card-cols {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  @media (max-width: 680px) {{ .card-cols {{ grid-template-columns: 1fr; }} }}
  .col h4 {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 6px; }}
  .col ul {{ list-style: none; font-size: 0.85rem; }}
  .col ul li {{ padding: 3px 0; border-bottom: 1px solid var(--border); }}
  .col ul li:last-child {{ border-bottom: none; }}
  .col ul li.empty {{ color: var(--muted); font-style: italic; }}
  .col ul li:not(.empty)::before {{ content: "• "; color: var(--accent); }}

  .transcript-toggle {{ margin-top: 14px; border-top: 1px solid var(--border); padding-top: 12px; }}
  .transcript-toggle button {{
    background: none; border: 1px solid var(--border); border-radius: 6px;
    color: var(--muted); cursor: pointer; font-size: 0.8rem; padding: 4px 12px;
  }}
  .transcript-toggle button:hover {{ color: var(--text); border-color: var(--accent); }}
  .transcript-body {{ margin-top: 12px; }}
  .transcript-body pre {{
    background: rgba(255,255,255,0.03); border-radius: 8px; font-size: 0.9rem;
    line-height: 1.7; color: #b0bcd4; padding: 16px;
    white-space: pre-wrap; word-break: break-word;
    max-height: 420px; overflow-y: auto;
    border: 1px solid var(--border);
  }}
  .transcript-body pre::-webkit-scrollbar {{ width: 8px; }}
  .transcript-body pre::-webkit-scrollbar-track {{ background: transparent; }}
  .transcript-body pre::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; min-height: 40px; }}
  .transcript-body pre::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

  .no-results {{ padding: 32px; color: var(--muted); text-align: center; }}
</style>
</head>
<body>

<header>
  <h1>{REPORT_TITLE}</h1>
  <p>{REPORT_SUBTITLE} &mdash; Generated {datetime.now().strftime('%B %d, %Y')}</p>
</header>

<div class="stats">
  <div class="stat"><div class="stat-value" id="meeting-count">{total_meetings}</div><div class="stat-label">Meetings</div></div>
  <div class="stat"><div class="stat-value">{total_decisions}</div><div class="stat-label">Decisions</div></div>
  <div class="stat"><div class="stat-value">{total_actions}</div><div class="stat-label">Action Items</div></div>
  <div class="stat"><div class="stat-value">{contentious}</div><div class="stat-label">Contentious</div></div>
</div>

<div class="controls">
  <input type="text" id="search" placeholder="Search meetings, topics, names..." oninput="applyFilters()">
  <button class="filter-btn active" onclick="setSentiment('all', this)">All</button>
  <button class="filter-btn" onclick="setSentiment('contentious', this)">Contentious</button>
  <button class="filter-btn" onclick="setSentiment('urgent', this)">Urgent</button>
  <button class="filter-btn" onclick="setSentiment('routine', this)">Routine</button>
  <button class="filter-btn" onclick="setSentiment('ceremonial', this)">Ceremonial</button>
  <button class="filter-btn" id="sort-btn" onclick="toggleSort()">Date ▼</button>
</div>

<div class="cards" id="cards">
{cards_html}
</div>

<script>
  let activeSentiment = 'all';

  function applyFilters() {{
    const q = document.getElementById('search').value.toLowerCase();
    let visible = 0;
    document.querySelectorAll('.card').forEach(card => {{
      const matchText      = !q || card.textContent.toLowerCase().includes(q);
      const matchSentiment = activeSentiment === 'all' || card.dataset.sentiment === activeSentiment;
      const show = matchText && matchSentiment;
      card.classList.toggle('hidden', !show);
      if (show) visible++;
    }});
    document.getElementById('meeting-count').textContent = visible;
    const nr = document.getElementById('no-results');
    if (nr) nr.remove();
    if (visible === 0) {{
      const div = document.createElement('div');
      div.id = 'no-results';
      div.className = 'no-results';
      div.textContent = 'No meetings match your filters.';
      document.getElementById('cards').appendChild(div);
    }}
  }}

  function setSentiment(sentiment, btn) {{
    activeSentiment = sentiment;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilters();
  }}

  function toggleTranscript(id, url) {{
    const body = document.getElementById('tr-' + id);
    const btn  = body.previousElementSibling;
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    btn.innerHTML = open ? btn.dataset.closed : btn.dataset.open;
    if (!open && url) {{
      const pre = document.getElementById('tr-text-' + id);
      if (pre && pre.textContent === 'Loading...') {{
        fetch(url)
          .then(r => r.text())
          .then(t => {{ pre.textContent = t; }})
          .catch(() => {{ pre.textContent = 'Could not load transcript.'; }});
      }}
    }}
  }}

  let sortAsc = false;
  function toggleSort() {{
    sortAsc = !sortAsc;
    const cards = document.getElementById('cards');
    const items = Array.from(cards.querySelectorAll('.card'));
    items.sort((a, b) => {{
      const da = a.dataset.date || '';
      const db = b.dataset.date || '';
      return sortAsc ? da.localeCompare(db) : db.localeCompare(da);
    }});
    items.forEach(c => cards.appendChild(c));
    document.getElementById('sort-btn').textContent = sortAsc ? 'Date ▲' : 'Date ▼';
  }}
</script>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"[report] Written to {REPORT_PATH} ({total_meetings} meetings)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Goshen Township Vimeo Pipeline")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Max videos to process this run (default: all unfinished)")
    parser.add_argument("--discover", action="store_true",
                        help="Discover new videos only, skip processing")
    parser.add_argument("--report",   action="store_true",
                        help="Regenerate HTML report only, no processing")
    parser.add_argument("--video-id", help="Process a single specific video ID")
    parser.add_argument("--cloud", action="store_true",
                        help="Transcribe via Groq → OpenAI only (no local fallback)")
    parser.add_argument("--local", action="store_true",
                        help="Transcribe locally only (no cloud)")
    parser.add_argument("--rename-files", action="store_true",
                        help="One-time migration: rename video_id-based files to title-based names")
    parser.add_argument("--recompress", action="store_true",
                        help="Compress oversized audio files (>25MB) that are missing transcripts")
    parser.add_argument("--fix-durations", action="store_true",
                        help="Backfill duration_sec for all audio files using ffprobe")
    parser.add_argument("--fix-json-durations", action="store_true",
                        help="Patch duration_sec into existing summary JSON files from DB")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status summary and exit")
    args = parser.parse_args()

    conn = init_db()

    if args.status:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM videos GROUP BY status ORDER BY count DESC"
        ).fetchall()
        total = sum(r[1] for r in rows)
        print(f"\n--- Pipeline Status ({total} total videos) ---")
        for status, count in rows:
            print(f"  {status:20} {count:4}")
        stuck = conn.execute(
            "SELECT title FROM videos WHERE status='audio_ready'"
        ).fetchall()
        if stuck:
            print(f"\n  Stuck at audio_ready (need --local or --cloud):")
            for r in stuck:
                print(f"    - {r[0]}")
        print()
        return

    if args.report:
        build_report(conn)
        return

    if args.rename_files:
        rename_files(conn)
        return

    if args.recompress:
        recompress_audio(conn)
        return

    if args.fix_durations:
        fix_durations(conn)
        return

    if args.fix_json_durations:
        fix_json_durations(conn)
        return

    discover_videos(conn)

    if args.discover:
        return

    # If running --local on a machine that doesn't have the audio files (e.g. Windows
    # pulling from CT), reset audio_ready videos with missing files back to discovered
    # so the download step re-fetches them from Vimeo in this same run.
    if args.local:
        rows = conn.execute(
            "SELECT video_id, audio_path FROM videos WHERE status='audio_ready'"
        ).fetchall()
        reset = 0
        for r in rows:
            if not r["audio_path"] or not Path(r["audio_path"]).exists():
                conn.execute(
                    "UPDATE videos SET status='discovered', audio_path=NULL, updated_at=? WHERE video_id=?",
                    (datetime.now(timezone.utc).isoformat(), r["video_id"])
                )
                reset += 1
        if reset:
            conn.commit()
            print(f"[local] Reset {reset} audio_ready video(s) with missing audio → discovered for re-download")

    if args.video_id:
        row = get_video(conn, args.video_id)
        if not row:
            print(f"Video ID {args.video_id} not found in manifest. Run --discover first.")
            sys.exit(1)
        videos = [row]
    else:
        videos = get_unfinished(conn, limit=args.limit)

    if not videos:
        print("Nothing to process — all discovered videos are fully summarized.")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    print(f"\nProcessing {len(videos)} video(s)...\n")

    stats = {"downloaded": 0, "skipped": 0, "groq": 0, "openai": 0, "local": 0, "summarized": 0}

    for row in videos:
        vid    = row["video_id"]
        status = row["status"]
        print(f"\n--- {row['title']} ({vid}) | status: {status} ---")

        if status == "discovered":
            if not download_audio(conn, row):
                stats["skipped"] += 1
                continue
            stats["downloaded"] += 1
            row = get_video(conn, vid)

        if row["status"] == "audio_ready":
            if args.local:
                ok = "local" if transcribe(conn, row) else None
            elif args.cloud:
                ok = transcribe_cloud(conn, row)
            else:
                ok = transcribe_cloud(conn, row) or ("local" if transcribe(conn, row) else None)
            if not ok:
                stats["skipped"] += 1
                continue
            stats[ok] += 1
            row = get_video(conn, vid)

        if row["status"] == "transcribed":
            if summarize(conn, row, client):
                stats["summarized"] += 1

    work_done = stats["downloaded"] + stats["groq"] + stats["openai"] + stats["local"] + stats["summarized"]
    if work_done > 0:
        print("\n--- Building report ---")
        build_report(conn)
    else:
        print("\n--- No changes — skipping report rebuild ---")

    print(f"""
--- Run Summary ---
  Downloaded:   {stats['downloaded']}
  Transcribed:  {stats['groq']} via Groq | {stats['openai']} via OpenAI | {stats['local']} local
  Summarized:   {stats['summarized']}
  Skipped:      {stats['skipped']}
""")


if __name__ == "__main__":
    main()
