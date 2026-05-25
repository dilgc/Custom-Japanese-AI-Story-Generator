import re
import subprocess

import ollama
import streamlit as st

from config import (
    CONTENT_POS,
    FALLBACK_MODEL,
    OLLAMA_MODEL,
    VOCAB_SAMPLE_SIZE,
)
from dictionary import cached_lookup
from tokenizer import tokenize_story
from ui_components import STORY_CSS, render_story_with_tooltips


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _gpu_info() -> list[dict]:
    """Return a list of dicts with GPU name and free/total VRAM in MiB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mib": int(parts[2]),
                    "free_mib": int(parts[3]),
                })
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _available_models() -> list[str]:
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def check_ollama_ready(model: str) -> tuple[bool, str]:
    try:
        available = _available_models()
        if any(model in m for m in available):
            return True, ""
        return False, (
            f"Model **{model}** not found in Ollama.\n\n"
            f"Run in a terminal: `ollama pull {model}`"
        )
    except Exception as e:
        return False, (
            f"Cannot connect to Ollama: {e}\n\n"
            "**Setup instructions:**\n"
            "1. Install Ollama from https://ollama.com\n"
            "2. Run: `ollama pull qwen3:8b`\n"
            "3. Restart this app."
        )


def strip_thinking(text: str) -> str:
    # Remove complete <think>...</think> blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned

    # Incomplete block: model ran out of tokens inside <think>.
    # Grab anything that appeared after the LAST </think>, or before <think>.
    if "</think>" in text:
        after = text.rsplit("</think>", 1)[-1].strip()
        if after:
            return after

    # No closing tag at all — everything was consumed by the thinking block.
    # Return empty so callers can show a useful error.
    return ""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_story_prompt(vocab_df, threshold_rank: int, topic: str | None = None) -> str:
    known_vocab = vocab_df[vocab_df["rank"] <= threshold_rank]
    content_words = known_vocab[known_vocab["pos"].apply(
        lambda p: any(p.startswith(cp) for cp in CONTENT_POS)
    )]

    mid_freq = content_words[content_words["rank"] > 100]
    sample_df = mid_freq.sample(n=min(VOCAB_SAMPLE_SIZE, len(mid_freq)))
    word_list = "、".join(sample_df["lemma"].tolist())

    topic_line = f"テーマ: {topic}" if topic else "テーマは自由に選んでください。"

    return (
        f"あなたは日本語学習者向けの短編小説作家です。\n\n"
        f"以下のルールに厳密に従って、10文程度の短い物語を書いてください：\n\n"
        f"1. 使用する語彙は、日本語の頻出語彙リストの上位{threshold_rank}語の範囲内に限定してください。\n"
        f"2. 以下は使用可能な語彙の例です。これらの語彙と同等かそれ以上に頻出な語彙のみを使ってください：\n"
        f"{word_list}\n"
        f"3. 文法は自然な日本語を使ってください。ただし、複雑すぎる構文は避けてください。\n"
        f"4. 物語は起承転結のある完結した話にしてください。\n"
        f"5. マークダウン記法（**太字**、*斜体*、# 見出しなど）は使わないでください。普通のテキストで書いてください。\n\n"
        f"{topic_line}\n\n"
        f"物語を書いてください："
    )


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that LLMs sometimes emit."""
    # Bold/italic: **word**, *word*, __word__, _word_
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{1,3}(.+?)_{1,3}",   r"\1", text, flags=re.DOTALL)
    # ATX headings: ## Title → Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def _use_thinking(model: str) -> bool:
    return True  # always use structured thinking; content arrives clean in msg.content


def _extract_japanese(text: str) -> str:
    """Pull out the Japanese story from a response that may contain English preamble.

    qwen3:4b with think=False writes reasoning in English, then produces Japanese
    story drafts as single long lines interspersed with English analysis.
    Strategy: collect all lines that are ≥75% Japanese, filter out character-count
    breakdowns, and return the longest one (= the most complete draft).
    """
    def jp_ratio(s: str) -> float:
        if not s:
            return 0.0
        jp = sum(1 for c in s if (
            "぀" <= c <= "ゟ" or  # hiragana
            "゠" <= c <= "ヿ" or  # katakana
            "一" <= c <= "鿿"     # kanji
        ))
        return jp / len(s)

    def is_count_line(s: str) -> bool:
        # "雨の日、 (3)" or ends with "(N)"
        return bool(re.search(r"\(\d+\)\s*$", s)) or \
               bool(re.search(r"[ぁ-んァ-ン一-龯] \(\d\)", s))

    lines = text.splitlines()

    # Collect lines that are strongly Japanese and not counting breakdowns
    candidates = [
        l.strip() for l in lines
        if len(l.strip()) >= 10
        and jp_ratio(l) >= 0.75
        and not is_count_line(l)
    ]

    if candidates:
        # Return the longest draft — it's the most complete version
        return max(candidates, key=len)

    # Fallback: grab from the first Japanese sentence onward
    m = re.search(r"[ぁ-んァ-ン一-龯ー]{3,}.{10,}[。！？]", text)
    if m:
        start = text.rfind("\n", 0, m.start())
        return text[max(0, start):].strip()

    return text


def generate_story(prompt: str, model: str, status_placeholder=None) -> str:
    """Stream the response from Ollama.

    qwen3:8b: thinking is ON — content arrives in message.content after the
    chain-of-thought finishes in message.thinking.

    qwen3:4b: thinking is OFF — the model still reasons in English inline, so we
    extract the Japanese paragraphs from the raw content afterward.
    """
    thinking_on = _use_thinking(model)

    messages = [{"role": "user", "content": prompt}]

    story_content = ""
    thinking_chars = 0

    for chunk in ollama.chat(
        model=model,
        messages=messages,
        think=thinking_on,
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 8192,   # 4b thinking can reach 4k+ tokens; leave room for story
            "num_ctx": 16384,
            "num_gpu": 99,
        },
        stream=True,
    ):
        msg = chunk.message
        story_content  += msg.content  or ""
        thinking_chars += len(msg.thinking or "")

        if status_placeholder is not None:
            if thinking_chars and not story_content:
                status_placeholder.caption(f"🤔 Thinking… ({thinking_chars:,} chars)")
            elif story_content:
                status_placeholder.caption("✍️ Writing story…")

    return story_content


# ---------------------------------------------------------------------------
# Story validation
# ---------------------------------------------------------------------------

def validate_story_vocab(story_tokens: list[dict], known_vocab_set: set) -> list[dict]:
    skip = {"記号", "補助記号", "助詞", "助動詞"}
    oov = []
    for token in story_tokens:
        if token["pos"] in skip:
            continue
        if token["lemma"] not in known_vocab_set and token["surface"] not in known_vocab_set:
            oov.append(token)
    return oov


# ---------------------------------------------------------------------------
# Model selector sidebar widget
# ---------------------------------------------------------------------------

_MODEL_INFO = {
    "qwen3:8b":  {"vram_mib": 5200, "label": "qwen3:8b  — better quality, ~5.2 GB VRAM"},
    "qwen3:4b":  {"vram_mib": 2800, "label": "qwen3:4b  — faster, ~2.8 GB VRAM (fits your GPU fully)"},
}


def _render_model_selector() -> str:
    """Render the model selector in the sidebar and return the chosen model name."""
    available = _available_models()

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Generation Model")

    gpus = _gpu_info()
    if gpus:
        g = gpus[0]
        free_gib = g["free_mib"] / 1024
        total_gib = g["total_mib"] / 1024
        st.sidebar.caption(f"GPU: {g['name']}")
        st.sidebar.caption(f"VRAM: {free_gib:.1f} GB free / {total_gib:.1f} GB total")

        # Warn if the selected model won't fit
        for model_id, info in _MODEL_INFO.items():
            if info["vram_mib"] > g["free_mib"] + 200:  # 200 MiB headroom
                pass  # warn per-option below

    options = []
    labels  = []
    for model_id, info in _MODEL_INFO.items():
        if any(model_id in m for m in available):
            options.append(model_id)
            label = info["label"]
            if gpus:
                free_mib = gpus[0]["free_mib"]
                fits = info["vram_mib"] <= free_mib + 200
                label += "  ✅ fits GPU" if fits else "  ⚠️ may spill to CPU"
            labels.append(label)

    if not options:
        st.sidebar.warning("No qwen3 models found. Run `ollama pull qwen3:8b`.")
        return OLLAMA_MODEL

    # Default to whatever is already selected, or pick the best fitting model
    if "selected_model" not in st.session_state:
        # Auto-select: prefer 8b if it fits, else 4b
        if gpus:
            free_mib = gpus[0]["free_mib"]
            st.session_state.selected_model = next(
                (m for m in options
                 if _MODEL_INFO.get(m, {}).get("vram_mib", 9999) <= free_mib + 200),
                options[0],
            )
        else:
            st.session_state.selected_model = options[0]

    current_idx = options.index(st.session_state.selected_model) if st.session_state.selected_model in options else 0

    chosen = st.sidebar.radio(
        "Model",
        options=options,
        format_func=lambda m: labels[options.index(m)],
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.selected_model = chosen

    if gpus and _MODEL_INFO.get(chosen, {}).get("vram_mib", 0) > gpus[0]["free_mib"] + 200:
        st.sidebar.info(
            f"Your GPU has {gpus[0]['free_mib']/1024:.1f} GB free. "
            f"This model needs ~{_MODEL_INFO[chosen]['vram_mib']/1024:.1f} GB, "
            f"so some layers will run on CPU — generation will be slower. "
            f"Pull `qwen3:4b` for full GPU speed."
        )

    return chosen


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def generate_story_ui(df) -> None:
    profile = st.session_state.profile
    threshold = profile["threshold_rank"]

    # Model selector lives in the sidebar
    chosen_model = _render_model_selector()

    st.header("📖 Story Generator")
    st.write(f"Generating stories using your top **{threshold:,}** vocabulary words.")

    # Ollama readiness check
    ready, msg = check_ollama_ready(chosen_model)
    if not ready:
        st.error("⚠️ Ollama is not ready.")
        st.markdown(msg)
        return
    if msg:
        st.info(msg)

    topic = st.text_input(
        "Story topic (optional — leave blank for a random topic):",
        placeholder="e.g., 猫の冒険、学校生活、旅行...",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        generate_btn = st.button("✨ Generate New Story", type="primary")
    with col2:
        regenerate = st.button("🔄 Regenerate")

    if generate_btn or regenerate:
        status = st.empty()
        status.caption("🤔 Thinking…")
        try:
            prompt = build_story_prompt(df, threshold, topic if topic else None)
            raw = generate_story(prompt, chosen_model, status_placeholder=status)
            # generate_story already returns only message.content (no think tags)
            story_text = _strip_markdown(raw)
            status.empty()
            if not story_text.strip():
                st.error("The model returned an empty story — try regenerating.")
            else:
                st.session_state.current_story = story_text
                st.session_state.current_story_topic = topic
        except Exception as e:
            status.empty()
            st.error(f"Generation failed: {e}")

    # Display story
    if "current_story" in st.session_state and st.session_state.current_story:
        story = st.session_state.current_story

        tokens = tokenize_story(story)

        unique_lemmas = {
            t["lemma"] for t in tokens
            if t["lemma"] and t["pos"] not in ("記号", "補助記号")
        }

        with st.spinner("Looking up words…"):
            lookups = {lemma: cached_lookup(lemma) for lemma in unique_lemmas}

        st.markdown(STORY_CSS, unsafe_allow_html=True)
        html = render_story_with_tooltips(tokens, lookups)
        st.markdown(
            f'<div class="story-container">{html}</div>',
            unsafe_allow_html=True,
        )
        st.caption("💡 Hover (or tap on mobile) any word to see its translation and grammar info.")

        # Stats expander
        with st.expander("Story Statistics"):
            content_tokens = [t for t in tokens if t["pos"] not in ("記号", "補助記号")]
            known_set = set(df[df["rank"] <= threshold]["lemma"])
            in_vocab = sum(1 for t in content_tokens if t["lemma"] in known_set)
            coverage = in_vocab / len(content_tokens) * 100 if content_tokens else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total tokens", len(content_tokens))
            c2.metric("Unique words", len(unique_lemmas))
            c3.metric("In your vocab", f"{coverage:.0f}%")

            oov = validate_story_vocab(tokens, known_set)
            if oov:
                st.caption(
                    f"{len(oov)} word(s) may be outside your threshold — "
                    "they're still clickable for lookup."
                )
