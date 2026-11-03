import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["NO_PROXY"] = "*"

try:
  import matplotlib
  import matplotlib.pyplot as plt
  import pandas as pd
except ModuleNotFoundError as exc:
  missing_package = getattr(exc, "name", "required package")
  raise SystemExit(
    f"Missing dependency: {missing_package}. Please install matplotlib and pandas to run visualizer.py."
  ) from exc

matplotlib.use("Agg")


BASE_DIR = Path(__file__).resolve().parent
VISIONS_DIR = BASE_DIR / "visions"
ANALYSIS_DIR = BASE_DIR / "analysis"
OUTPUT_PATH = ANALYSIS_DIR / "character_trend.png"

matplotlib.rcParams["font.sans-serif"] = [
  "PingFang SC",
  "Hiragino Sans GB",
  "Arial Unicode MS",
  "Noto Sans CJK SC",
  "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

STOP_WORDS = {
  "the", "and", "with", "from", "that", "this", "into", "over", "wearing",
  "wears", "wear", "look", "looks", "looking", "appearance", "stage",
  "character", "style", "visual", "outfit", "their", "there", "while",
  "where", "which", "when", "his", "her", "hers", "him", "she", "he",
  "they", "them", "its", "for", "has", "have", "had", "are", "was", "were",
  "but", "not", "too", "very", "more", "most", "like", "just", "then",
  "than", "also", "still", "into", "across", "about", "around", "under",
  "after", "before", "through", "between", "during", "includes", "include",
  "including", "reflecting", "suggesting", "showing", "stage", "name",
  "signature", "accessory", "unknown", "baseline", "appearance",
}


def clamp_sharp_tongue_level(value: Any, default: int = 5) -> int:
  try:
    level = int(value)
  except (TypeError, ValueError):
    level = default
  return max(1, min(10, level))


def load_json_documents(directory: Path) -> list[dict[str, Any]]:
  if not directory.exists():
    raise FileNotFoundError(f"visions directory not found: {directory}")

  documents: list[dict[str, Any]] = []
  for json_path in sorted(directory.glob("*.json")):
    with json_path.open("r", encoding="utf-8") as file:
      payload = json.load(file)
    if isinstance(payload, dict):
      payload["_source_path"] = str(json_path)
      documents.append(payload)

  if not documents:
    raise FileNotFoundError(f"No JSON files found in visions directory: {directory}")
  return documents


def build_stage_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []

  for document in documents:
    source_path = Path(str(document.get("_source_path", "")))
    novel_id = str(document.get("novel_id") or source_path.stem or "unknown")
    root_characters = document.get("characters", [])

    if isinstance(root_characters, list) and root_characters:
      for character_entry in root_characters:
        rows.extend(extract_character_rows(character_entry, novel_id))
      continue

    fallback_entry = {
      "name": document.get("character_name") or "Unknown",
      "visual_stages": document.get("visual_stages"),
      "stages": document.get("stages"),
    }
    rows.extend(extract_character_rows(fallback_entry, novel_id))

  return rows


def extract_character_rows(character_entry: dict[str, Any], novel_id: str) -> list[dict[str, Any]]:
  character_name = str(character_entry.get("name") or "Unknown").strip() or "Unknown"
  stages = character_entry.get("visual_stages")
  if not isinstance(stages, list) or not stages:
    stages = character_entry.get("stages", [])

  if not isinstance(stages, list):
    return []

  rows: list[dict[str, Any]] = []
  for order, stage in enumerate(stages, start=1):
    if not isinstance(stage, dict):
      continue

    stage_name = str(stage.get("stage_name") or f"Stage {order}").strip()
    outfit_description = str(
      stage.get("outfit_description")
      or stage.get("description")
      or ""
    ).strip()

    key_markers = stage.get("key_visual_markers", [])
    if not outfit_description and isinstance(key_markers, list):
      outfit_description = ", ".join(str(marker).strip() for marker in key_markers if str(marker).strip())

    iconic_item = str(stage.get("iconic_item") or "").strip()
    if not iconic_item and isinstance(key_markers, list):
      iconic_item = ", ".join(str(marker).strip() for marker in key_markers[:2] if str(marker).strip())

    rows.append(
      {
        "novel_id": novel_id,
        "character": character_name,
        "stage_order": int(stage.get("stage_index") or order),
        "stage_name": stage_name,
        "sharp_tongue_level": clamp_sharp_tongue_level(stage.get("sharp_tongue_level", 5)),
        "outfit_description": outfit_description,
        "iconic_item": iconic_item or "Signature Accessory",
      }
    )

  return rows


def build_stage_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
  if not rows:
    raise ValueError("No stage data could be extracted from the JSON files.")

  frame = pd.DataFrame(rows)
  frame["character_key"] = frame["novel_id"] + " / " + frame["character"]
  frame = frame.sort_values(["novel_id", "character", "stage_order", "stage_name"]).reset_index(drop=True)
  return frame


def tokenize_theme_text(text: str) -> list[str]:
  lowered = text.lower()
  english_tokens = re.findall(r"[a-z][a-z0-9'-]{1,}", lowered)
  chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", text)
  tokens = english_tokens + chinese_tokens

  cleaned: list[str] = []
  for token in tokens:
    if token in STOP_WORDS:
      continue
    if token.isdigit():
      continue
    cleaned.append(token)
  return cleaned


def build_theme_counter(frame: pd.DataFrame) -> Counter[str]:
  counter: Counter[str] = Counter()
  for _, row in frame.iterrows():
    text = f"{row['outfit_description']} {row['iconic_item']}"
    counter.update(tokenize_theme_text(text))
  return counter


def plot_sharp_tongue_timeline(ax: plt.Axes, frame: pd.DataFrame) -> None:
  color_map = matplotlib.colormaps["tab20"].resampled(max(frame["character_key"].nunique(), 1))

  for color_index, (character_key, group) in enumerate(frame.groupby("character_key", sort=False)):
    ordered = group.sort_values("stage_order")
    ax.plot(
      ordered["stage_order"],
      ordered["sharp_tongue_level"],
      marker="o",
      linewidth=2,
      label=character_key,
      color=color_map(color_index),
    )

    for _, row in ordered.iterrows():
      ax.annotate(
        str(row["stage_name"]),
        (row["stage_order"], row["sharp_tongue_level"]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
      )

  ax.set_title("Sharp Tongue Timeline")
  ax.set_xlabel("Stage Order")
  ax.set_ylabel("Sharp Tongue Level")
  ax.set_ylim(1, 10)
  ax.set_yticks(range(1, 11))
  ax.grid(True, linestyle="--", alpha=0.3)
  ax.legend(loc="upper left", fontsize=8, frameon=False)


def plot_visual_tag_cloud(ax: plt.Axes, counter: Counter[str], top_n: int = 24) -> None:
  ax.set_title("Visual Tag Cloud")
  ax.axis("off")

  if not counter:
    ax.text(0.5, 0.5, "No visual keywords found", ha="center", va="center", fontsize=16)
    return

  items = counter.most_common(top_n)
  max_count = items[0][1]
  min_count = items[-1][1]
  rows = math.ceil(len(items) / 4)
  colors = matplotlib.colormaps["viridis"].resampled(max(len(items), 1))

  for index, (word, count) in enumerate(items):
    grid_row = index // 4
    grid_col = index % 4
    x = (grid_col + 0.5) / 4
    y = 1 - (grid_row + 0.5) / rows

    if max_count == min_count:
      font_size = 20
    else:
      font_size = 12 + (count - min_count) * 18 / (max_count - min_count)

    rotation = 0 if index % 3 else 12
    ax.text(
      x,
      y,
      word,
      fontsize=font_size,
      rotation=rotation,
      ha="center",
      va="center",
      color=colors(index),
      transform=ax.transAxes,
      alpha=0.9,
    )


def render_analysis(frame: pd.DataFrame, output_path: Path) -> Path:
  ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
  theme_counter = build_theme_counter(frame)

  figure, axes = plt.subplots(2, 1, figsize=(16, 12), constrained_layout=True)
  plot_sharp_tongue_timeline(axes[0], frame)
  plot_visual_tag_cloud(axes[1], theme_counter)

  figure.suptitle("Character Visual and Behavioral Analysis", fontsize=18)
  figure.savefig(output_path, dpi=220, bbox_inches="tight")
  plt.close(figure)
  return output_path


def main() -> Path:
  documents = load_json_documents(VISIONS_DIR)
  rows = build_stage_rows(documents)
  frame = build_stage_frame(rows)
  return render_analysis(frame, OUTPUT_PATH)


if __name__ == "__main__":
  result_path = main()
  print(f"Saved analysis chart to {result_path}")
