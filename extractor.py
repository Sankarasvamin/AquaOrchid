import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

from ingestor import NovelIngestor

os.environ["NO_PROXY"] = "*"


class CharacterExtractor:
  def __init__(
    self,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 180,
    max_retries: int = 4,
    chunk_size: int = 20000,
    overlap: int = 500,
  ) -> None:
    base_dir = Path(__file__).resolve().parent
    self._load_env_file(base_dir / ".env")
    resolved_api_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not resolved_api_key:
      raise ValueError(
        "Missing OPENAI_API_KEY. Please add it to AquaOrchid/.env "
        "as OPENAI_API_KEY=your_key or export it in shell."
      )

    self.ingestor = NovelIngestor(chunk_size=chunk_size, overlap=overlap)
    self.model = model
    self.api_key = resolved_api_key
    self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).strip() or "https://api.openai.com/v1/chat/completions"
    self.timeout = timeout
    self.max_retries = max_retries

  def _load_env_file(self, env_path: Path) -> None:
    if not env_path.exists():
      return

    for line in env_path.read_text(encoding="utf-8").splitlines():
      stripped = line.strip()
      if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
      key, value = stripped.split("=", 1)
      key = key.strip()
      value = value.strip().strip("'\"")
      if key and key not in os.environ:
        os.environ[key] = value

  def _parse_json(self, raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
      text = re.sub(r"^```(?:json)?\n", "", text)
      text = re.sub(r"\n```$", "", text)

    try:
      return json.loads(text)
    except json.JSONDecodeError:
      match = re.search(r"\{[\s\S]*\}", text)
      if match:
        return json.loads(match.group(0))
      raise

  def _ask_json(self, prompt: str) -> Dict[str, Any]:
    payload = {
      "model": self.model,
      "messages": [{"role": "user", "content": prompt}],
      "temperature": 0,
    }
    last_error: Exception | None = None

    for attempt in range(1, self.max_retries + 1):
      req = request.Request(
        self.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
          "Authorization": f"Bearer {self.api_key}",
          "Content-Type": "application/json",
          "User-Agent": "AquaOrchid/1.0",
          "Connection": "close",
        },
        method="POST",
      )
      try:
        with request.urlopen(req, timeout=self.timeout) as response:
          body = response.read().decode("utf-8")

        data = json.loads(body)
        choices = data.get("choices", [])
        if not choices:
          raise RuntimeError(f"LLM response missing choices: {body}")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
          text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
          ]
          content = "\n".join(part for part in text_parts if part)

        return self._parse_json(str(content))
      except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < self.max_retries:
          last_error = RuntimeError(f"LLM request failed: HTTP {exc.code} {detail}")
          time.sleep(min(2 ** (attempt - 1), 8))
          continue
        raise RuntimeError(f"LLM request failed: HTTP {exc.code} {detail}") from exc
      except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, OSError) as exc:
        last_error = exc
        if attempt < self.max_retries:
          time.sleep(min(2 ** (attempt - 1), 8))
          continue
        break

    raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error

  def _select_character_scan_chunks(self, chunks: List[str], target_count: int = 8) -> List[str]:
    if len(chunks) <= target_count:
      return chunks

    selected_indexes: List[int] = []
    last_index = len(chunks) - 1
    for i in range(target_count):
      index = round(i * last_index / (target_count - 1))
      if index not in selected_indexes:
        selected_indexes.append(index)
    return [chunks[index] for index in selected_indexes]

  def identify_main_characters(self, chunks: List[str]) -> List[str]:
    scan_chunks = self._select_character_scan_chunks(chunks, target_count=8)
    scan_text = "\n\n".join(scan_chunks)
    prompt = f'''You are extracting character information from a web novel.

Read the text and identify the top 5 to 8 most prominent named characters by narrative focus.
Rules:
- Prefer recurring named characters, not generic roles.
- Prioritize characters who receive visual description, dialogue focus, or plot-central presence.
- Keep the output deduplicated.
Return valid JSON only, with this schema:
{{
  "characters": ["Character Name 1", "Character Name 2", "Character Name 3", "Character Name 4", "Character Name 5"]
}}

Text:
{scan_text}
'''
    result = self._ask_json(prompt)
    names = result.get("characters", [])
    if not isinstance(names, list):
      return []
    deduped: List[str] = []
    for name in names:
      cleaned = str(name).strip()
      if cleaned and cleaned not in deduped:
        deduped.append(cleaned)
    return deduped[:8]

  def extract_descriptions(self, character: str, chunks: List[str]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
      prompt = f'''Based on this text, describe {character}'s current visual presentation in rich detail.

Focus on:
- physique, age impression, face, hair, eyes, expression
- clothing pieces, uniform details, fabrics, colors, silhouette
- accessories, props, bandages, injuries, supernatural traits, posture
- any visual changes compared with earlier appearances in the story

Writing rules:
- If there are visual clues, write 2 to 5 full sentences.
- Aim for at least 60 Chinese characters or 40 English words when clues exist.
- Be concrete and descriptive, not vague.
- If the character is present but visual clues are weak, still summarize whatever can be inferred from the passage.

Return valid JSON only:
{{
  "character": "{character}",
  "chunk_index": {idx},
  "description": "...",
  "has_visual_signal": true
}}

If no clear visual clues are present, set description to an empty string and has_visual_signal to false.

Text:
{chunk}
'''
      result = self._ask_json(prompt)
      description = str(result.get("description", "")).strip()
      has_visual_signal = bool(result.get("has_visual_signal", False))
      if has_visual_signal and description and len(description) < 60:
        description = self._expand_description(character, idx, chunk, description)
      records.append(
        {
          "chunk_index": idx,
          "description": description,
          "has_visual_signal": has_visual_signal,
        }
      )
    return records

  def _expand_description(self, character: str, chunk_index: int, chunk: str, seed_description: str) -> str:
    prompt = f'''Rewrite and expand this visual description for {character}.

Current short description:
{seed_description}

Requirements:
- Expand it into 3 to 6 concrete sentences.
- Mention visible body traits, clothing, colors, materials, accessories, wounds, posture, and mood when supported.
- Keep it faithful to the text and avoid empty filler.
- Target at least 100 Chinese characters or 70 English words.

Return valid JSON only:
{{
  "description": "expanded description"
}}

Chunk index: {chunk_index}
Text:
{chunk}
'''
    result = self._ask_json(prompt)
    expanded = str(result.get("description", "")).strip()
    return expanded or seed_description

  def _clamp_sharp_tongue_level(self, value: Any, default: int = 5) -> int:
    try:
      level = int(value)
    except (TypeError, ValueError):
      level = default
    return max(1, min(10, level))

  def _infer_iconic_item(self, outfit_description: str) -> str:
    lowered = outfit_description.lower()
    candidates = [
      "glasses",
      "sword",
      "knife",
      "box cutter",
      "stapler",
      "umbrella",
      "bag",
      "handbag",
      "backpack",
      "ribbon",
      "hat",
      "scarf",
      "uniform",
      "coat",
      "jacket",
      "necklace",
      "earrings",
      "ring",
      "cane",
      "book",
      "phone",
      "camera",
    ]
    for candidate in candidates:
      if candidate in lowered:
        return candidate.title()
    return "Signature Accessory"

  def _fallback_visual_stages(self, character: str, descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible_descriptions = [
      str(item.get("description", "")).strip()
      for item in descriptions
      if item.get("has_visual_signal") and str(item.get("description", "")).strip()
    ]
    merged_description = " ".join(visible_descriptions)[:1200].strip()
    if not merged_description:
      merged_description = (
        f"{character} has limited explicit visual detail in the sampled EPUB chunks. "
        "Preserve a grounded, story-faithful outfit concept with a clear silhouette, "
        "readable accessories, and restrained facial attitude."
      )

    return [
      {
        "stage_name": "Baseline Appearance",
        "sharp_tongue_level": 5,
        "outfit_description": merged_description,
        "iconic_item": self._infer_iconic_item(merged_description),
      }
    ]

  def _normalize_visual_stages(
    self,
    character: str,
    raw_stages: Any,
    descriptions: List[Dict[str, Any]],
    max_stages: int,
  ) -> List[Dict[str, Any]]:
    if not isinstance(raw_stages, list):
      return self._fallback_visual_stages(character, descriptions)

    normalized: List[Dict[str, Any]] = []
    for index, stage in enumerate(raw_stages[:max_stages], start=1):
      if not isinstance(stage, dict):
        continue

      stage_name = str(stage.get("stage_name", "")).strip() or f"Stage {index}"
      outfit_description = str(
        stage.get("outfit_description")
        or stage.get("visual_summary")
        or stage.get("description")
        or ""
      ).strip()
      if not outfit_description:
        continue

      normalized.append(
        {
          "stage_name": stage_name,
          "sharp_tongue_level": self._clamp_sharp_tongue_level(stage.get("sharp_tongue_level", 5)),
          "outfit_description": outfit_description,
          "iconic_item": str(
            stage.get("iconic_item")
            or self._infer_iconic_item(outfit_description)
          ).strip() or "Signature Accessory",
        }
      )

    if normalized:
      return normalized
    return self._fallback_visual_stages(character, descriptions)

  def cluster_stages(self, character: str, descriptions: List[Dict[str, Any]], max_stages: int = 10) -> List[Dict[str, Any]]:
    payload = json.dumps(descriptions, ensure_ascii=False)
    prompt = f'''You are analyzing a novel character's visual and behavioral evolution.

Character: {character}
Data: {payload}

Rules:
1) Compare descriptions across chunk indices.
2) Prioritize stage breaks that align with chapter transitions in the EPUB.
3) If explicit chapter markers are not visible, infer chapter-to-chapter transitions from sequential chunk order, scene resets, and narrative time jumps.
4) Only create a new stage when a chapter transition or a clear visual/behavioral shift justifies it.
5) Keep chronological order.
6) sharp_tongue_level must be an integer from 1 to 10, reflecting how cutting, sarcastic, blunt, or verbally aggressive the character is in that stage.
7) outfit_description must be a detailed visual description suitable for Stable Diffusion prompting, including clothing pieces, silhouette, materials, colors, grooming, expression, and posture when supported.
8) iconic_item should be the most memorable prop, accessory, or garment associated with the stage.
9) Return at most {max_stages} stages.

Return valid JSON only with schema:
{{
  "visual_stages": [
    {{
      "stage_name": "...",
      "sharp_tongue_level": 1,
      "outfit_description": "...",
      "iconic_item": "..."
    }}
  ]
}}
'''
    result = self._ask_json(prompt)
    raw_stages = result.get("visual_stages")
    if raw_stages is None:
      raw_stages = result.get("stages", [])
    return self._normalize_visual_stages(character, raw_stages, descriptions, max_stages)

  def generate_sd_prompt(self, character: str, stage: Dict[str, Any]) -> str:
    stage_name = str(stage.get("stage_name", "Unnamed Stage"))
    sharp_tongue_level = self._clamp_sharp_tongue_level(stage.get("sharp_tongue_level", 5))
    outfit_description = str(stage.get("outfit_description", "")).strip()
    iconic_item = str(stage.get("iconic_item", "")).strip()

    prompt = f'''Create one Stable Diffusion prompt for the character stage below.

Character: {character}
Stage Name: {stage_name}
Sharp Tongue Level: {sharp_tongue_level}/10
Outfit Description: {outfit_description}
Iconic Item: {iconic_item}

Style requirements:
- VOFAN style
- vivid colors
- sharp shadows
- high-end fabric textures (silk/leather)
- monogatari aesthetic
- faithfully translate the outfit description into concrete prompt language
- include the iconic item clearly
- body language and expression should subtly reflect the sharp tongue level

Output valid JSON only:
{{
  "sd_prompt": "single-line prompt"
}}
'''
    result = self._ask_json(prompt)
    return str(result.get("sd_prompt", "")).strip()

  def build_manifest(self, epub_path: str, output_path: str = "character_manifest.json") -> Dict[str, Any]:
    all_chunks = self.ingestor.get_chunks(epub_path)
    target_chunks = all_chunks[:12]

    if not target_chunks:
      raise ValueError("No text chunks were extracted from the EPUB.")

    characters = self.identify_main_characters(target_chunks)
    manifest: Dict[str, Any] = {
      "source_epub": epub_path,
      "total_chunks": len(all_chunks),
      "processed_chunks": len(target_chunks),
      "characters": [],
    }

    for character in characters:
      descriptions = self.extract_descriptions(character, target_chunks)
      visual_stages = self.cluster_stages(character, descriptions, max_stages=10)

      enriched_stages: List[Dict[str, Any]] = []
      for stage in visual_stages:
        sd_prompt = self.generate_sd_prompt(character, stage)
        stage_with_prompt = dict(stage)
        stage_with_prompt["sd_prompt"] = sd_prompt
        enriched_stages.append(stage_with_prompt)

      manifest["characters"].append(
        {
          "name": character,
          "descriptions": descriptions,
          "visual_stages": enriched_stages,
          "stages": enriched_stages,
        }
      )

    output_file = Path(output_path)
    output_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
  epub_file = "monogatari01.epub"
  extractor = CharacterExtractor(model="gpt-4o")
  try:
    result = extractor.build_manifest(epub_file, output_path="character_manifest.json")
    print(
      f"Done. Characters: {len(result.get('characters', []))}, "
      f"output: character_manifest.json"
    )
  except Exception as exc:
    print(f"Extraction failed: {exc}")
