import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from xml.etree import ElementTree
from zipfile import ZipFile


class NovelIngestor:
  def __init__(self, chunk_size: int = 20000, overlap: int = 0) -> None:
    if chunk_size <= 0:
      raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0:
      raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
      raise ValueError("overlap must be smaller than chunk_size.")

    self.chunk_size = chunk_size
    self.overlap = overlap

  def _decode_content(self, raw_content: bytes) -> str:
    if not raw_content:
      return ""

    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5")
    for encoding in encodings:
      try:
        return raw_content.decode(encoding)
      except UnicodeDecodeError:
        continue
    return raw_content.decode("utf-8", errors="ignore")

  def _clean_text(self, text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

  def _get_spine_documents(self, epub_file: ZipFile) -> List[str]:
    container_xml = epub_file.read("META-INF/container.xml")
    container_root = ElementTree.fromstring(container_xml)
    rootfile = container_root.find(".//{*}rootfile")
    if rootfile is None:
      raise ValueError("Invalid EPUB: META-INF/container.xml is missing rootfile.")

    opf_path = rootfile.attrib.get("full-path", "").strip()
    if not opf_path:
      raise ValueError("Invalid EPUB: rootfile full-path is empty.")

    opf_root = ElementTree.fromstring(epub_file.read(opf_path))
    manifest = {}
    for item in opf_root.findall(".//{*}manifest/{*}item"):
      item_id = item.attrib.get("id", "").strip()
      href = item.attrib.get("href", "").strip()
      if item_id and href:
        manifest[item_id] = href

    opf_dir = Path(opf_path).parent
    ordered_docs: List[str] = []
    for itemref in opf_root.findall(".//{*}spine/{*}itemref"):
      item_id = itemref.attrib.get("idref", "").strip()
      href = manifest.get(item_id)
      if not href:
        continue
      resolved = (opf_dir / href).as_posix()
      if resolved.lower().endswith((".xhtml", ".html", ".htm")):
        ordered_docs.append(resolved)

    if ordered_docs:
      return ordered_docs

    fallback = [
      name for name in epub_file.namelist()
      if name.lower().endswith((".xhtml", ".html", ".htm"))
    ]
    return sorted(fallback)

  def _html_to_text(self, raw_html: str) -> str:
    class _TextExtractor(HTMLParser):
      def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self.ignored_depth = 0
        self.block_tags = {
          "address", "article", "aside", "blockquote", "br", "div", "dd",
          "dl", "dt", "figcaption", "figure", "footer", "form", "h1", "h2",
          "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol",
          "p", "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
          "tr", "ul",
        }
        self.ignored_tags = {"script", "style"}

      def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in self.ignored_tags:
          self.ignored_depth += 1
          return
        if self.ignored_depth:
          return
        if tag.lower() in self.block_tags:
          self.parts.append("\n")

      def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.ignored_tags:
          self.ignored_depth = max(0, self.ignored_depth - 1)
          return
        if self.ignored_depth:
          return
        if tag.lower() in self.block_tags:
          self.parts.append("\n")

      def handle_data(self, data: str) -> None:
        if self.ignored_depth:
          return
        if data:
          self.parts.append(data)

    parser = _TextExtractor()
    parser.feed(raw_html)
    parser.close()
    text = unescape("".join(parser.parts))
    return self._clean_text(text)

  def _looks_like_front_matter(self, doc_name: str, plain_text: str) -> bool:
    lowered_name = doc_name.lower()
    if any(marker in lowered_name for marker in ("cover", "titlepage", "contents", "toc", "message", "logo", "backcover")):
      return True

    sample = plain_text[:500]
    front_matter_hits = [
      "封面",
      "制作信息",
      "目录",
      "插画：",
      "译者：",
      "作者：",
    ]
    return sum(marker in sample for marker in front_matter_hits) >= 2

  def _extract_text(self, epub_path: str) -> str:
    segments: List[str] = []

    with ZipFile(epub_path) as epub_file:
      for doc_name in self._get_spine_documents(epub_file):
        html_text = self._decode_content(epub_file.read(doc_name))
        plain = self._html_to_text(html_text)
        if plain and not self._looks_like_front_matter(doc_name, plain):
          segments.append(plain)

    merged = "\n".join(segments)
    return self._clean_text(merged)

  def _chunk_text(self, text: str) -> List[str]:
    if not text:
      return []

    chunks: List[str] = []
    step = self.chunk_size - self.overlap
    start = 0
    length = len(text)

    while start < length:
      end = min(start + self.chunk_size, length)
      chunk = text[start:end].strip()
      if chunk:
        chunks.append(chunk)
      start += step

    return chunks

  def get_chunks(self, epub_path: str) -> List[str]:
    epub_file = Path(epub_path)
    if not epub_file.exists():
      raise FileNotFoundError(f"EPUB file not found: {epub_path}")
    if epub_file.suffix.lower() != ".epub":
      raise ValueError("Input file must be an .epub file.")

    full_text = self._extract_text(str(epub_file))
    return self._chunk_text(full_text)


if __name__ == "__main__":
  dummy_path = "monogatari01.epub"
  ingestor = NovelIngestor(chunk_size=20000, overlap=500)
  try:
    chunks = ingestor.get_chunks(dummy_path)
    print(f"Total chunks: {len(chunks)}")
    if chunks:
      preview = chunks[0][:200]
      print(f"First chunk preview:\n{preview}")
  except Exception as exc:
    print(f"Ingestion failed: {exc}")
