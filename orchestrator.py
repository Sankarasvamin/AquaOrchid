import json
import logging
import re
from pathlib import Path
from typing import Any

from extractor import CharacterExtractor
from ingestor import NovelIngestor


class PipelineOrchestrator:
  def __init__(
    self,
    source_dir: str | Path = ".",
    output_dir: str | Path = "visions",
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    chunk_size: int = 20000,
    overlap: int = 500,
  ) -> None:
    self.base_dir = Path(__file__).resolve().parent
    self.source_dir = self._resolve_path(source_dir)
    self.output_dir = self._resolve_path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)

    self.ingestor = NovelIngestor(chunk_size=chunk_size, overlap=overlap)
    self.extractor = CharacterExtractor(
      model=model,
      api_key=api_key,
      base_url=base_url,
      chunk_size=chunk_size,
      overlap=overlap,
    )
    self.logger = self._build_logger()

  def _resolve_path(self, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (self.base_dir / path)

  def _build_logger(self) -> logging.Logger:
    logger = logging.getLogger("PipelineOrchestrator")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
      return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(self.base_dir / "orchestrator_errors.log", encoding="utf-8")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

  def _scan_epub_files(self) -> list[Path]:
    files = sorted(self.source_dir.glob("*.epub"))
    self.logger.info("扫描到 EPUB 文件数量: %s", len(files))
    return files

  def _chunk_epub_with_ingestor(self, epub_path: Path) -> list[str]:
    chunks = self.ingestor.get_chunks(str(epub_path))
    if not chunks:
      raise ValueError(f"EPUB 分块结果为空: {epub_path}")
    return chunks

  def _extract_novel_id(self, source_path: Path) -> str:
    match = re.search(r"\d+", source_path.stem)
    if match:
      return match.group(0)

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_")
    return safe_stem or "unknown"

  def _consolidate_stages(self, character: str, descriptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = json.dumps(descriptions, ensure_ascii=False)
    prompt = f'''You are a character visual evolution analyst.

Character: {character}
Chunked observations: {payload}

Task:
1) Consolidate the character's visual evolution across the whole novel.
2) Output EXACTLY 3 to 10 typical stages.
3) Keep strict chronological order.
4) Focus especially on eccentric fashion, high-end textures (silk, rugged leather), and specific accessories.
5) If data is sparse, still infer reasonable stage boundaries from timeline and subtle clues.
6) Each stage description must be detailed and specific, ideally 3 to 6 sentences.

Return valid JSON only:
{{
  "stages": [
    {{
      "stage_index": 1,
      "stage_name": "...",
      "chunk_span": [1, 12],
      "description": "detailed visual description",
      "key_visual_markers": ["...", "..."]
    }}
  ]
}}
'''
    result = self.extractor._ask_json(prompt)
    stages = result.get("stages", [])

    if isinstance(stages, list) and 3 <= len(stages) <= 10:
      return self._normalize_stage_descriptions(stages)

    repair_prompt = f'''The previous stage output is invalid because stage count must be 3 to 10.

Character: {character}
Original observations: {payload}
Previous output: {json.dumps(result, ensure_ascii=False)}

Regenerate valid JSON with EXACTLY 3 to 10 stages.
Keep chronological order and preserve strong fashion/material details.
Return JSON only with key "stages".
'''
    repaired = self.extractor._ask_json(repair_prompt)
    repaired_stages = repaired.get("stages", [])

    if isinstance(repaired_stages, list) and 3 <= len(repaired_stages) <= 10:
      return self._normalize_stage_descriptions(repaired_stages)

    return self._fallback_stages(descriptions)

  def _normalize_stage_descriptions(self, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for stage in stages:
      item = dict(stage)
      description = str(item.get("description", "")).strip()
      if description:
        item["description"] = description
      normalized.append(item)
    return normalized

  def _fallback_stages(self, descriptions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not descriptions:
      return [
        {
          "stage_index": 1,
          "stage_name": "Initial Presence",
          "chunk_span": [1, 1],
          "description": "Visual data is limited; retain a restrained baseline profile.",
          "key_visual_markers": ["baseline silhouette", "minimal accessory clues"],
        },
        {
          "stage_index": 2,
          "stage_name": "Transitional Signals",
          "chunk_span": [1, 1],
          "description": "Subtle shifts are implied by narrative tone and sparse cues.",
          "key_visual_markers": ["possible styling shift", "uncertain texture emphasis"],
        },
        {
          "stage_index": 3,
          "stage_name": "Mature Image",
          "chunk_span": [1, 1],
          "description": "A final stabilized image inferred from available context.",
          "key_visual_markers": ["stable outfit logic", "final accessory profile"],
        },
      ]

    total = len(descriptions)
    boundaries = [0, total // 3, (2 * total) // 3, total]
    stages: list[dict[str, Any]] = []

    for idx in range(3):
      start = boundaries[idx]
      end = boundaries[idx + 1]
      block = descriptions[start:end]
      block_text = " ".join(
        str(item.get("description", "")).strip()
        for item in block
        if item.get("has_visual_signal") and str(item.get("description", "")).strip()
      )
      if not block_text:
        block_text = "Visual cues are sparse in this section; infer transitional styling continuity."

      span_start = start + 1
      span_end = max(span_start, end)
      stages.append(
        {
          "stage_index": idx + 1,
          "stage_name": f"Stage {idx + 1}",
          "chunk_span": [span_start, span_end],
          "description": block_text[:1200],
          "key_visual_markers": [
            "eccentric fashion contour",
            "high-end texture emphasis",
            "signature accessory hints",
          ],
        }
      )

    return stages

  def _generate_sd_prompt(self, character: str, stage: dict[str, Any]) -> str:
    prompt = f'''Create one Stable Diffusion prompt for this character evolution stage.

Character: {character}
Stage Data: {json.dumps(stage, ensure_ascii=False)}

Requirements:
- VOFAN x Nisio Isin visual mood
- eccentric fashion language
- high-end textures: silk, rugged leather, metal accessories
- cinematic framing, sharp shadows, rich color depth
- include concrete clothing pieces and accessories

Return valid JSON only:
{{
  "sd_prompt": "single-line prompt"
}}
'''
    result = self.extractor._ask_json(prompt)
    return str(result.get("sd_prompt", "")).strip()

  def _analyze_character(self, character_name: str, descriptions: list[dict[str, Any]]) -> dict[str, Any]:
    stages = self._consolidate_stages(character_name, descriptions)

    stages = stages[:10]
    if len(stages) < 3:
      stages = self._fallback_stages(descriptions)

    enriched_stages: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
      stage_data = dict(stage)
      stage_data["stage_index"] = index
      sd_prompt = self._generate_sd_prompt(character_name, stage_data)
      stage_data["sd_prompt"] = sd_prompt
      enriched_stages.append(stage_data)

    return {
      "name": character_name,
      "descriptions": descriptions,
      "stages": enriched_stages,
    }

  def _analyze_book(self, novel_id: str, source_path: Path, chunks: list[str]) -> Path:
    self.logger.info("开始分析 | novel_id=%s | chunks=%s", novel_id, len(chunks))

    characters = self.extractor.identify_main_characters(chunks)
    if len(characters) < 5:
      raise ValueError(f"识别到的角色数量不足 5 个: {source_path.name} -> {characters}")

    selected_characters = characters[:5]
    self.logger.info("识别角色: %s", " | ".join(selected_characters))

    character_payloads: list[dict[str, Any]] = []
    for character_name in selected_characters:
      self.logger.info("开始角色分析: %s", character_name)
      descriptions = self.extractor.extract_descriptions(character_name, chunks)
      character_payloads.append(self._analyze_character(character_name, descriptions))

    output = {
      "novel_id": novel_id,
      "source_file": source_path.name,
      "character_count": len(character_payloads),
      "character_name": character_payloads[0]["name"],
      "stages": character_payloads[0]["stages"],
      "characters": character_payloads,
    }

    output_path = self.output_dir / f"{novel_id}_vision.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    self.logger.info("导出完成: %s", output_path)
    return output_path

  def run_file(self, epub_path: str | Path) -> Path:
    source_path = self._resolve_path(epub_path)
    novel_id = self._extract_novel_id(source_path)
    chunks = self._chunk_epub_with_ingestor(source_path)
    return self._analyze_book(novel_id, source_path, chunks)

  def run(self) -> list[Path]:
    epub_files = self._scan_epub_files()
    if not epub_files:
      self.logger.warning("目录中没有 EPUB 文件: %s", self.source_dir)
      return []

    outputs: list[Path] = []
    for epub_path in epub_files:
      novel_id = self._extract_novel_id(epub_path)

      try:
        chunks = self._chunk_epub_with_ingestor(epub_path)
      except Exception as exc:
        self.logger.error(
          "分块失败，跳过该 EPUB | novel_id=%s | file=%s | error=%s",
          novel_id,
          epub_path.name,
          exc,
          exc_info=True,
        )
        continue

      try:
        output_path = self._analyze_book(novel_id, epub_path, chunks)
        outputs.append(output_path)
      except Exception as exc:
        self.logger.error(
          "分析失败，跳过该 EPUB | novel_id=%s | file=%s | error=%s",
          novel_id,
          epub_path.name,
          exc,
          exc_info=True,
        )
        continue

    self.logger.info("流水线结束，成功输出 %s 个文件。", len(outputs))
    return outputs


if __name__ == "__main__":
  orchestrator = PipelineOrchestrator(model="gpt-4o")
  result = orchestrator.run_file("monogatari.epub")
  print(f"Done. Generated {result.name}.")
