# Local Government Meeting Archive

Automatically discovers, transcribes, and summarizes public meeting videos from a Vimeo account. Generates a searchable, filterable static HTML report — no server required.

Built for Goshen Township (Clermont County, Ohio) but designed to work with any local government that publishes meeting videos publicly on Vimeo.

---

## What It Does

1. **Discovers** new videos from a public Vimeo account using yt-dlp
2. **Downloads** audio-only MP3 (auto-compressed to mono 32kbps if over 25MB)
3. **Transcribes** using a 3-tier fallback chain:
   - Groq `whisper-large-v3` (fast, free tier)
   - OpenAI `whisper-1` (fallback if Groq rate-limits)
   - Local `faster-whisper large-v3-turbo` (offline fallback)
4. **Summarizes** using Claude (Anthropic API) — extracts decisions, action items, financials, attendees, topics, and sentiment
5. **Generates** a static `report.html` with searchable, filterable meeting cards and on-demand transcript viewer
6. **Deletes** audio after transcription — the archive stays lean (~10MB for 200+ meetings)

All progress is tracked in a local SQLite database. The pipeline is fully resumable — stop and restart at any point.

---

## Report Features

- Filter by sentiment: All / Contentious / Urgent / Routine / Ceremonial
- Full-text search across all meetings
- Sort by date (ascending / descending)
- Meeting count updates dynamically with active filter
- Expandable transcript viewer per meeting (loaded on demand)
- Expandable attendee list per meeting
- Direct Vimeo link on each card
- Stats bar: total meetings, decisions, action items, contentious count

---

## Setup

### 1. Install dependencies

```bash
pip install yt-dlp faster-whisper anthropic groq openai python-dotenv pyyaml
```

Install [ffmpeg](https://ffmpeg.org/download.html) and note the path to its `bin/` folder.

### 2. Configure API keys

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
FFMPEG_PATH=C:\path\to\ffmpeg\bin
```

### 3. Configure your township

Edit `config.yaml` with your local government's details:

```yaml
township:
  name: "Your Township Name"
  county: "Your County"
  state: "Your State"
  address: "123 Main St, Your Town, ST 00000"
  vimeo_url: "https://vimeo.com/yourtownshiphandle"

officials:
  - "Jane Smith (Mayor)"
  - "John Doe (Clerk)"

local_vocabulary:
  roads:
    - "Main Street"
    - "Route 50"
  places:
    - "Town Hall"
    - "County Planning Commission"
  procedural:
    - "motion carries"
    - "second the motion"
    - "adjourn"

report:
  title: "Your Township Meeting Archive"
  subtitle: "Your County, Your State &mdash; Public Meetings"
```

The `officials` and `local_vocabulary` fields seed the Whisper transcription model with proper nouns specific to your area — this significantly improves accuracy for names, roads, and procedural terms that speech models commonly mangle.

---

## Usage

```bash
# Process all unfinished videos (full fallback chain)
python vimeo_pipeline.py

# Process up to N videos
python vimeo_pipeline.py --limit 5

# Cloud transcription only (Groq → OpenAI, no local model)
python vimeo_pipeline.py --cloud

# Local transcription only (offline, CPU-based)
python vimeo_pipeline.py --local

# Discover new videos only, no processing
python vimeo_pipeline.py --discover

# Process one specific video by ID
python vimeo_pipeline.py --video-id 123456789

# Regenerate the HTML report from existing data
python vimeo_pipeline.py --report

# Show pipeline status summary
python vimeo_pipeline.py --status
```

---

## Viewing the Report

The report uses `fetch()` to load transcripts on demand, so it requires a local server to work correctly:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/report.html`.

When hosted on GitHub Pages or any static host, it works without any extra setup.

---

## Project Structure

```
vimeo_pipeline.py     # Main pipeline script
config.yaml           # Township-specific configuration
report.html           # Generated report (open in browser)
data/
  videos.db           # SQLite manifest — tracks pipeline state per video
  transcripts/        # Plain text transcripts (also served by the report)
  summaries/          # JSON summaries per meeting
```

---

## Requirements

- Python 3.10+
- ffmpeg
- API keys: Anthropic, Groq, OpenAI (OpenAI is fallback only)
- For local transcription: `faster-whisper` + a CPU (GPU not required)

---

## Notes

- Tested against public Vimeo accounts. No authentication required.
- Cloud transcription (`--cloud`) uses no local GPU/CPU beyond running the script itself.
- The pipeline processes videos oldest-first and skips any already completed.
- Audio files are automatically deleted after transcription to keep storage minimal.
