# AquaOrchid

AquaOrchid is a local pipeline for studying how fictional characters' visual descriptions evolve across an EPUB novel.

## Workflow

1. Read user-provided EPUB files in spine order.
2. Convert chapter HTML into normalized text and overlapping chunks.
3. Use a configured language model to extract visual evidence.
4. Consolidate observations into chronological character stages.
5. Produce structured JSON/Markdown results and a local trend visualization.

## Public-data boundary

This repository contains only processing code. Novels, generated character profiles, images, session files, logs, and download automation are excluded. Use only EPUB files that you are legally allowed to process.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Place authorized EPUB files in a local working directory and configure `OPENAI_API_KEY`. The main pipeline is implemented in `orchestrator.py`; `visualizer.py` reads generated JSON files from `visions/`.

## Files

- `ingestor.py`: EPUB spine parsing and text chunking
- `extractor.py`: structured LLM extraction with retry handling
- `orchestrator.py`: end-to-end stage consolidation
- `visualizer.py`: aggregate character-stage visualization

