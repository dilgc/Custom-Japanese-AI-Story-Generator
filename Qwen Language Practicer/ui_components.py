from config import SKIP_POS

STORY_CSS = """
<style>
.story-container {
    font-size: 22px;
    line-height: 2.2;
    font-family: "Noto Sans JP", "Hiragino Sans", "Yu Gothic", sans-serif;
    padding: 24px;
    background: #fafafa;
    border-radius: 8px;
    margin: 10px 0;
}
.word {
    position: relative;
    cursor: pointer;
    border-bottom: 1px dotted #aaa;
    transition: background 0.15s;
}
.word:hover, .word:focus {
    background: #e8f4fd;
    border-radius: 3px;
    outline: none;
}
.tooltip {
    display: none;
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #1e293b;
    color: #f1f5f9;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.7;
    min-width: 240px;
    max-width: 340px;
    z-index: 1000;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    text-align: left;
    white-space: normal;
    pointer-events: none;
}
.word:hover .tooltip,
.word:focus .tooltip {
    display: block;
}
.tooltip-reading {
    font-size: 15px;
    font-weight: bold;
    color: #93c5fd;
    display: block;
    margin-bottom: 4px;
}
.tooltip-dict {
    color: #94a3b8;
    font-size: 12px;
}
.tooltip-form {
    color: #fbbf24;
    font-size: 12px;
}
.tooltip-def {
    margin-top: 6px;
    border-top: 1px solid #334155;
    padding-top: 6px;
}
.punct { cursor: default; }
</style>
"""

_PUNCT_CHARS = set("。、！？「」『』（）【】…―・〜\n\r\t ")


def _escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_story_with_tooltips(tokens: list[dict], lookups: dict[str, dict]) -> str:
    html_parts = []

    for token in tokens:
        surface = token["surface"]
        lemma = token["lemma"] or surface

        if token["pos"] in SKIP_POS or all(c in _PUNCT_CHARS for c in surface):
            html_parts.append(f'<span class="punct">{_escape(surface)}</span>')
            continue

        lookup = lookups.get(lemma, {})
        definitions = lookup.get("definitions", [])
        reading = token.get("reading", "")

        tooltip_lines = []

        # Header: surface + reading
        if reading and reading != surface:
            tooltip_lines.append(
                f'<span class="tooltip-reading">{_escape(surface)} ({_escape(reading)})</span>'
            )
        else:
            tooltip_lines.append(
                f'<span class="tooltip-reading">{_escape(surface)}</span>'
            )

        # Dictionary form if different
        if lemma != surface:
            tooltip_lines.append(
                f'<span class="tooltip-dict">dict: {_escape(lemma)}</span>'
            )

        # Conjugation info
        if token.get("cForm"):
            tooltip_lines.append(
                f'<span class="tooltip-form">form: {_escape(token["cForm_en"] or token["cForm"])}</span>'
            )
        if token.get("cType"):
            tooltip_lines.append(
                f'<span class="tooltip-form">type: {_escape(token["cType_en"] or token["cType"])}</span>'
            )

        # Definitions
        if definitions:
            defs_html = "<br>".join(
                f'{i+1}. {_escape(d)}' for i, d in enumerate(definitions[:3])
            )
            tooltip_lines.append(f'<span class="tooltip-def">{defs_html}</span>')
        else:
            tooltip_lines.append('<span class="tooltip-def"><em>no definition found</em></span>')

        tooltip_inner = "".join(tooltip_lines)

        html_parts.append(
            f'<span class="word" tabindex="0">{_escape(surface)}'
            f'<span class="tooltip">{tooltip_inner}</span></span>'
        )

    return "".join(html_parts)
