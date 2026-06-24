"""
Citation Tracer - Source Attribution Engine
Parses LLM answers, extracts citation markers, matches to source documents,
and generates rich source cards with exact text spans.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document


@dataclass
class Citation:
    """A single citation linking a reference marker to its source document."""
    marker: int               # Citation number, e.g. [1] -> 1
    source_index: int         # Index in the sources list (0-based)
    filename: str
    content_preview: str      # First ~100 chars of the source chunk
    full_content: str         # Full source chunk text
    page: Optional[int] = None
    chunk_id: str = ""
    relevance: str = ""       # "direct" | "related" | "context"


@dataclass
class CitationResult:
    """Parsed result from an LLM answer with citations."""
    answer: str                          # Original answer text
    citations: List[Citation] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    cited_markers: List[int] = field(default_factory=list)     # Which [N] markers were actually used
    uncited_sources: List[int] = field(default_factory=list)   # Sources provided but not cited

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


class CitationTracer:
    """
    Traces citations in LLM-generated answers back to source documents.

    Capabilities:
    - Parse [1], [2] markers from answer text
    - Match markers to source documents
    - Generate rich Markdown/HTML source cards
    - Validate citation accuracy
    - Highlight relevant sentence spans
    """

    # Regex to find citation markers like [1], [2,3], [1-3]
    CITATION_PATTERN = re.compile(r"\[(\d+(?:[-,]\d+)*)\]")

    # Max chars to show in source preview
    PREVIEW_LENGTH = 200

    def parse(self, answer: str, sources: List[Dict[str, Any]]) -> CitationResult:
        """
        Parse citation markers from an LLM answer and link them to sources.

        Args:
            answer: The full LLM-generated answer text.
            sources: List of source dicts (from RAGResult.sources).

        Returns:
            CitationResult with parsed citations and metadata.
        """
        result = CitationResult(answer=answer, sources=sources)

        # 1. Extract all citation markers from the answer
        markers = self._extract_markers(answer)
        if not markers:
            # No inline citations found -> all sources are "uncited"
            result.uncited_sources = list(range(len(sources)))
            return result

        result.cited_markers = markers

        # 2. Match each marker to a source
        for marker in markers:
            source_idx = marker - 1  # [1] -> sources[0]
            if 0 <= source_idx < len(sources):
                src = sources[source_idx]
                citation = Citation(
                    marker=marker,
                    source_index=source_idx,
                    filename=src.get("filename", "unknown"),
                    content_preview=src.get("content", "")[:self.PREVIEW_LENGTH],
                    full_content=src.get("content", ""),
                    page=src.get("page"),
                    chunk_id=src.get("chunk_id", ""),
                    relevance=self._classify_relevance(answer, src.get("content", "")),
                )
                result.citations.append(citation)

        # 3. Identify uncited sources
        cited_indices = {c.source_index for c in result.citations}
        result.uncited_sources = [
            i for i in range(len(sources)) if i not in cited_indices
        ]

        return result

    def format_markdown(self, result: CitationResult) -> str:
        """
        Generate a Markdown-formatted source reference section.

        Output format:
            ---
            **References:**

            [1] **filename.pdf** (Page 3)
            > Relevant excerpt from the document...

            [2] **FAQ.md**
            > Another relevant excerpt...
        """
        if not result.citations:
            return ""

        lines = ["---", "**References:**", ""]
        for c in result.citations:
            page_str = f" (Page {c.page})" if c.page else ""
            lines.append(f"[{c.marker}] **{c.filename}**{page_str}")

            # Show a relevant text span
            snippet = self._find_best_snippet(result.answer, c.full_content)
            if snippet:
                lines.append(f"> {snippet}")
            else:
                lines.append(f"> {c.content_preview[:150]}...")
            lines.append("")

        return "\n".join(lines)

    def format_html(self, result: CitationResult) -> str:
        """
        Generate an HTML source card for UI rendering.

        Suitable for Streamlit/Gradio display.
        """
        if not result.citations:
            return ""

        parts = ['<div class="citations">', '<h4>References</h4>']
        for c in result.citations:
            page_str = f" (Page {c.page})" if c.page else ""
            parts.append(f"""
            <div class="citation-item" style="margin-bottom:12px;padding:8px;background:#f5f5f5;border-radius:4px;">
                <strong>[{c.marker}]</strong> <em>{c.filename}</em>{page_str}
                <blockquote style="margin:4px 0;color:#555;border-left:3px solid #ccc;padding-left:8px;">
                    {c.content_preview[:200]}...
                </blockquote>
            </div>""")
        parts.append('</div>')
        return "\n".join(parts)

    def validate(self, result: CitationResult) -> Dict[str, Any]:
        """
        Validate citation accuracy.

        Returns:
            {
                "total_citations": int,
                "valid": bool,
                "issues": [str, ...],
                "coverage": float  # % of sources that are cited
            }
        """
        issues = []
        total_markers = len(result.cited_markers)

        # Check for out-of-range citations
        for marker in result.cited_markers:
            if marker < 1 or marker > len(result.sources):
                issues.append(f"Citation [{marker}] references non-existent source")

        # Check for duplicate citations
        if len(result.cited_markers) != len(set(result.cited_markers)):
            issues.append("Duplicate citation markers found")

        # Check coverage
        coverage = 0.0
        if result.sources:
            coverage = result.citation_count / len(result.sources)

        return {
            "total_citations": result.citation_count,
            "total_markers": total_markers,
            "valid": len(issues) == 0,
            "issues": issues,
            "coverage": round(coverage, 2),
            "uncited_sources": result.uncited_sources,
        }

    # =================================================================
    #  Internal helpers
    # =================================================================

    def _extract_markers(self, text: str) -> List[int]:
        """Extract all unique citation marker numbers from answer text."""
        markers = set()
        for match in self.CITATION_PATTERN.finditer(text):
            group = match.group(1)
            # Handle ranges like [1-3], lists like [2,4], and single [5]
            parts = group.split(",")
            for part in parts:
                part = part.strip()
                if "-" in part:
                    # Range: "1-3" -> [1, 2, 3]
                    try:
                        start, end = part.split("-", 1)
                        for n in range(int(start.strip()), int(end.strip()) + 1):
                            markers.add(n)
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        markers.add(int(part))
                    except ValueError:
                        pass
        return sorted(markers)

    def _classify_relevance(self, answer: str, source_content: str) -> str:
        """
        Classify citation relevance: direct / related / context.
        Simple heuristic: check keyword overlap between answer and source.
        """
        if not source_content:
            return "context"

        answer_words = set(answer.lower().split())
        source_words = set(source_content.lower().split())
        overlap = len(answer_words & source_words)

        if overlap > 10:
            return "direct"
        elif overlap > 3:
            return "related"
        return "context"

    def _find_best_snippet(self, answer: str, source_content: str, max_len: int = 200) -> str:
        """
        Find the most relevant snippet from the source content
        by finding overlapping phrases with the answer.
        """
        if not source_content:
            return ""

        # Simple approach: find common n-grams and return surrounding text
        answer_sentences = re.split(r"[。.!！?\n]", answer)
        source_sentences = re.split(r"[。.!！?\n]", source_content)

        best_sentence = ""
        best_overlap = 0

        for src_sent in source_sentences:
            src_sent = src_sent.strip()
            if not src_sent or len(src_sent) < 10:
                continue

            src_words = set(src_sent)
            for ans_sent in answer_sentences:
                ans_sent = ans_sent.strip()
                if not ans_sent:
                    continue
                ans_words = set(ans_sent)
                overlap = len(src_words & ans_words)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = src_sent

        if best_sentence:
            return best_sentence[:max_len] + ("..." if len(best_sentence) > max_len else "")

        # Fallback: first 200 chars
        return source_content[:max_len] + "..."


# ========== Convenience functions ==========

def trace_citations(answer: str, sources: List[Dict[str, Any]]) -> CitationResult:
    """One-liner: parse and trace citations in an answer."""
    return CitationTracer().parse(answer, sources)


def format_citation_card(sources: List[Dict[str, Any]], fmt: str = "markdown") -> str:
    """
    Format source documents as a citation card (without an answer).
    Useful for displaying sources alongside the retrieved context.
    """
    if fmt == "html":
        parts = ['<div class="sources"><h4>Sources</h4>']
        for i, src in enumerate(sources, start=1):
            filename = src.get("filename", "unknown")
            page = src.get("page")
            page_str = f" (Page {page})" if page else ""
            content = src.get("content", "")[:200]
            parts.append(f"""
            <div style="margin-bottom:8px;padding:6px;background:#f9f9f9;border-radius:4px;">
                <strong>[{i}]</strong> <em>{filename}</em>{page_str}
                <p style="margin:4px 0;color:#666;font-size:0.9em;">{content}...</p>
            </div>""")
        parts.append('</div>')
        return "\n".join(parts)
    else:
        lines = ["**Sources:**", ""]
        for i, src in enumerate(sources, start=1):
            filename = src.get("filename", "unknown")
            page = src.get("page")
            page_str = f" (Page {page})" if page else ""
            content = src.get("content", "")[:150]
            lines.append(f"[{i}] **{filename}**{page_str}")
            lines.append(f"    > {content}...")
            lines.append("")
        return "\n".join(lines)
