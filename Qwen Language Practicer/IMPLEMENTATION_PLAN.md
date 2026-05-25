# Japanese Vocabulary Threshold Estimator & Adaptive Story Generator

## Complete Implementation Plan for Claude Code

---

## 1. Project Overview

Build a Streamlit application that:

1. Adaptively quizzes a user to estimate their **vocabulary frequency threshold** — the frequency rank below which they recognize ~90% of words in a Japanese frequency list.
2. Generates a **short story using only words within that threshold**, powered by a fully local LLM.
3. Makes every word in the generated story **clickable**, revealing its English translation, part of speech, conjugation form, and contextual usage.
4. **Persists** the user's assessment results to disk (JSON) so they survive app restarts, with a button to retake the assessment at any time.

---

## 2. Source Data: NLT1.40 Frequency List

The Excel file `NLT1.40_freq_list.xlsx` lives at `experiment/data/NLT1.40_freq_list.xlsx`. Its columns are:

| Column | Japanese Header | Content | Example |
|--------|----------------|---------|---------|
| A | レマ | Lemma (dictionary form of the word) | 食べる |
| B | 品詞 | Part of speech | 動詞-自立 |
| C | 読み | Reading in katakana | タベル |
| D | 頻度 | Raw frequency count (descending) | 6,636,525 |

**Important characteristics:**

- Rows are pre-sorted by frequency (highest first). The row index is the frequency rank.
- The list contains punctuation (記号), proper noun categories (【地域】, 【人名】, 【組織】, 【数字】), and function words alongside content words.
- Part-of-speech values use MeCab/UniDic conventions with subcategories separated by hyphens (e.g., `動詞-自立`, `動詞-非自立`, `動詞-接尾`).
- Some entries have empty 読み fields (typically for punctuation and category markers).

### Data Preprocessing (at first launch / cached)

On startup, load the Excel file and build a clean DataFrame. This is a one-time operation cached with `@st.cache_data`:

```
1. Load with openpyxl (pandas reads .xlsx via openpyxl).
2. Assign a 1-based "rank" column from the row order.
3. Filter OUT rows where 品詞 == "記号" (punctuation/symbols).
4. Filter OUT rows where レマ starts with "【" (category headers like 【地域】, 【数字】).
5. Strip whitespace from all string columns.
6. Store the result as a global DataFrame: columns [rank, lemma, pos, reading, freq].
```

The filtered list is what the quiz and story generator operate on. Expect roughly 200,000–400,000 usable entries after filtering (the original file contains "millions" of rows including inflected forms and symbols).

### Performance Note on Large Files

If the file truly has millions of rows, loading it every time Streamlit reruns would be slow. The solution:

```python
@st.cache_data
def load_frequency_list():
    df = pd.read_excel("experiment/data/NLT1.40_freq_list.xlsx", engine="openpyxl")
    df.columns = ["lemma", "pos", "reading", "freq"]
    df = df[df["pos"] != "記号"]
    df = df[~df["lemma"].str.startswith("【", na=False)]
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
```

If loading is still too slow (>10 seconds), convert the Excel to a Parquet file on first run and load from Parquet thereafter. Parquet reads are 10–50x faster than Excel for large datasets:

```python
import os
PARQUET_PATH = "experiment/data/freq_list.parquet"

@st.cache_data
def load_frequency_list():
    if os.path.exists(PARQUET_PATH):
        return pd.read_parquet(PARQUET_PATH)
    df = pd.read_excel("experiment/data/NLT1.40_freq_list.xlsx", engine="openpyxl")
    # ... filtering as above ...
    df.to_parquet(PARQUET_PATH, index=False)
    return df
```

---

## 3. Adaptive Vocabulary Assessment Algorithm

### Goal

Find the frequency rank R such that the user knows approximately 90% of words with rank ≤ R and less than 90% of words beyond R. This is the user's "vocabulary frontier."

### Why Not Simple Binary Search

A naive binary search (present one word at rank N, go higher if known, lower if unknown) is fragile because vocabulary knowledge is noisy — users may not know a common word due to domain gaps, or may know a rare word from personal interest. A single incorrect answer would drastically misplace the boundary.

### Chosen Algorithm: Batched Adaptive Binary Search with Bayesian Smoothing

This approach combines the efficiency of binary search with the noise tolerance of sampling multiple words per level:

```
PHASE 1 — COARSE BRACKETING (4–6 rounds, ~5 words each)
─────────────────────────────────────────────────────────
1. Define the search space: rank_low = 1, rank_high = total_vocab_count.
2. Pick the midpoint: rank_mid = (rank_low + rank_high) // 2.
3. Sample 5 words uniformly at random from the band [rank_mid - 250, rank_mid + 250].
   - Exclude 記号, function words (助詞, 助動詞) from the sample, since these
     are grammatical and don't test vocabulary knowledge well.
   - Prefer content words: 名詞, 動詞-自立, 形容詞, 副詞, 形容動詞.
4. Present the 5 words to the user. For each word, the user clicks "I know this" or
   "I don't know this."
5. Calculate the hit rate for this batch (e.g., 4/5 = 80%).
6. If hit_rate >= 0.8: the user likely knows words at this level. Set rank_low = rank_mid.
   If hit_rate < 0.5: the user likely doesn't. Set rank_high = rank_mid.
   If 0.5 <= hit_rate < 0.8: the frontier is nearby. Narrow the band but don't jump fully.
     Set rank_high = rank_mid + (rank_high - rank_mid) // 3.
7. Repeat until rank_high - rank_low < 2000. This takes about 5–6 rounds.

PHASE 2 — FINE ESTIMATION (3–5 rounds, ~5 words each)
──────────────────────────────────────────────────────
8. Within the narrowed band [rank_low, rank_high], sample 5 words per round.
9. Track cumulative hit rate across all Phase 2 samples.
10. After each round, compute the estimated threshold using Bayesian updating:
    - Model: P(know | rank) = sigmoid(a * (threshold - rank))
    - Use the accumulated responses to find the threshold where P(know) = 0.9.
    - Simpler alternative: weighted linear interpolation between the lowest rank
      where hit_rate >= 0.9 and the highest rank where hit_rate < 0.9.
11. Stop when the standard error of the estimate < 200 ranks, or after 5 rounds.

TOTAL: 7–11 rounds × 5 words = 35–55 words shown. Takes 2–4 minutes.
```

### Simplified Implementation (Recommended for v1)

For the initial build, use the simpler weighted interpolation approach rather than full Bayesian/IRT modeling. Full IRT requires careful parameter calibration that isn't worth the complexity for v1:

```python
def estimate_threshold(responses: list[dict]) -> int:
    """
    responses: [{"rank": 5000, "known": True}, ...]
    Returns estimated frequency rank threshold.
    """
    # Group responses into rank bands of width 500
    bands = defaultdict(list)
    for r in responses:
        band = (r["rank"] // 500) * 500
        bands[band].append(r["known"])

    # Find where hit rate crosses below 0.9
    for band_start in sorted(bands.keys()):
        hit_rate = sum(bands[band_start]) / len(bands[band_start])
        if hit_rate < 0.7:  # Using 0.7 as the band threshold (accounts for noise)
            return band_start

    return max(bands.keys()) + 500  # User knows everything tested
```

### Word Sampling Strategy

When sampling words for a quiz round, apply these filters to the frequency list:

```python
QUIZZABLE_POS = {"名詞", "動詞-自立", "形容詞", "副詞", "形容動詞", "連体詞"}

def sample_quiz_words(df, center_rank, n=5, band_width=500):
    low = max(1, center_rank - band_width)
    high = min(len(df), center_rank + band_width)
    band = df[(df["rank"] >= low) & (df["rank"] <= high)]

    # Filter to content words only
    content = band[band["pos"].isin(QUIZZABLE_POS) |
                   band["pos"].str.startswith("動詞-自立") |
                   band["pos"].str.startswith("形容")]

    if len(content) < n:
        content = band[band["pos"] != "助詞"]  # Fallback: anything except particles

    return content.sample(n=min(n, len(content)))
```

### Quiz UI Design

Each round shows 5 words in a clean card layout:

```
┌─────────────────────────────────────────────┐
│  Round 3 of ~8                              │
│                                             │
│  Do you know these words?                   │
│                                             │
│  ┌──────────┐  ┌──────────┐                 │
│  │  食べる   │  │  走る    │                 │
│  │  ✅  ❌  │  │  ✅  ❌  │                 │
│  └──────────┘  └──────────┘                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  美しい   │  │  概念    │  │  挑戦     │  │
│  │  ✅  ❌  │  │  ✅  ❌  │  │  ✅  ❌  │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│                                             │
│           [ Submit Round ]                  │
└─────────────────────────────────────────────┘
```

Use `st.columns` for layout. Each word card is rendered with `st.button` for "Know" / "Don't Know" or with two-column buttons per card. Words display in large font (use `st.markdown` with custom CSS for the kanji display at ~32px).

---

## 4. Local LLM: Story Generation

### Model Selection: Qwen3 8B via Ollama

After evaluating available options for local Japanese story generation, the recommendation is:

| Option | VRAM | Japanese Quality | Speed | Verdict |
|--------|------|-----------------|-------|---------|
| **Qwen3 8B (Q4_K_M)** | **~4.6 GB** | **Excellent** | **~42 tok/s** | **Recommended** |
| Qwen3 14B (Q4_K_M) | ~8.3 GB | Better | ~28 tok/s | If GPU allows |
| Qwen3 30B-A3B (MoE) | ~17 GB | Best | ~35 tok/s | If 24GB GPU |
| Qwen 2.5 7B | ~4.5 GB | Very good | ~45 tok/s | Fallback |

**Why Qwen3 8B:**
- The Qwen family has the strongest Japanese language support among open-weight models at every size tier. It was trained with native Japanese tokenization, processing Japanese text 30–40% more token-efficiently than Llama or Mistral models.
- At Q4_K_M quantization, it runs on virtually any GPU with 6+ GB VRAM, or even on CPU with 8+ GB RAM (slower but functional at ~8 tok/s).
- Thinking mode can be disabled (`/nothink`) for fast, direct story generation without reasoning overhead.
- Apache 2.0 license — no restrictions on use.

**Fallback for CPU-only machines:** Qwen3 4B at Q4 needs only ~2.8 GB and still produces coherent Japanese text. Quality drops noticeably for complex grammar but works for simple stories aimed at lower-level learners.

### Ollama Setup (Prerequisite — document for user)

The app should check for Ollama availability on startup and display clear setup instructions if missing:

```python
import subprocess

def check_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return "qwen3" in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
```

If Ollama is not found, show:
```
⚠️ Ollama is required but not running.
1. Install from https://ollama.com
2. Run: ollama pull qwen3:8b
3. Restart this app.
```

### Story Generation Prompt Engineering

The prompt must be carefully constructed to constrain the LLM to use only words within the user's vocabulary range. This is the most critical prompt in the system:

```python
def build_story_prompt(vocab_df, threshold_rank, topic=None):
    """
    vocab_df: the full frequency DataFrame
    threshold_rank: user's estimated vocabulary threshold
    topic: optional user-chosen topic
    """
    # Extract the vocabulary list the user knows (top N by frequency)
    known_vocab = vocab_df[vocab_df["rank"] <= threshold_rank]

    # Get a representative sample of content words to include in the prompt
    # (sending all 10,000+ words would overflow the context window)
    content_words = known_vocab[known_vocab["pos"].isin(CONTENT_POS)]

    # Sample ~300 words weighted toward mid-frequency (more interesting than top-100)
    mid_freq = content_words[content_words["rank"] > 100]
    sample = mid_freq.sample(n=min(300, len(mid_freq)))
    word_list = "、".join(sample["lemma"].tolist())

    topic_line = f"テーマ: {topic}" if topic else "テーマは自由に選んでください。"

    prompt = f"""あなたは日本語学習者向けの短編小説作家です。

以下のルールに厳密に従って、300〜500字の短い物語を書いてください：

1. 使用する語彙は、日本語の頻出語彙リストの上位{threshold_rank}語の範囲内に限定してください。
2. 以下は使用可能な語彙の例です。これらの語彙と同等かそれ以上に頻出な語彙のみを使ってください：
{word_list}
3. 文法は自然な日本語を使ってください。ただし、複雑すぎる構文は避けてください。
4. 漢字には必要に応じてふりがなを付けないでください（ふりがな不要）。
5. 物語は起承転結のある完結した話にしてください。

{topic_line}

物語を書いてください："""

    return prompt
```

### Calling Ollama from Python

```python
import ollama

def generate_story(prompt: str, model: str = "qwen3:8b") -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 1024,    # Max tokens for story
            "num_ctx": 4096,        # Context window
        },
        # Disable thinking mode for direct output
        # Qwen3 thinking mode is toggled via /nothink in the prompt
        # or by setting think=False in newer ollama versions
    )
    return response["message"]["content"]
```

**Install the Python client:** `pip install ollama`

### Handling Thinking Mode

Qwen3 outputs `<think>...</think>` blocks by default. Two approaches:

1. **Strip them post-generation:** Parse the response and remove everything between `<think>` and `</think>` tags.
2. **Disable at the prompt level:** Add `/nothink` at the start of the user message (supported in newer Ollama versions), or use a Modelfile that sets `PARAMETER think false`.

Recommended: Strip post-generation (more reliable across Ollama versions):

```python
import re

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
```

---

## 5. Morphological Analysis & Word Segmentation

### Library: fugashi + unidic-lite

**fugashi** is the standard Python wrapper for MeCab (the industry-standard Japanese morphological analyzer). It segments Japanese text into words and provides part-of-speech, lemma (dictionary form), reading, and conjugation information for each token.

**Install:**
```bash
pip install fugashi unidic-lite
```

- `fugashi` provides the MeCab wrapper with prebuilt wheels for all platforms.
- `unidic-lite` is a compact (47 MB) dictionary based on UniDic 2.1.2. It provides all the morphological features needed. The full `unidic` (770 MB) offers slightly better accuracy but the improvement is negligible for this use case.

### Tokenization Pipeline

```python
from fugashi import Tagger

tagger = Tagger()  # Uses unidic-lite by default

def tokenize_story(text: str) -> list[dict]:
    """
    Tokenize a Japanese story into words with full morphological information.
    Returns a list of token dicts.
    """
    tokens = []
    for word in tagger(text):
        token = {
            "surface": str(word),              # Surface form as it appears
            "lemma": word.feature.lemma,        # Dictionary form
            "pos": word.feature.pos1,           # Major POS (名詞, 動詞, etc.)
            "pos2": word.feature.pos2,          # Sub-POS (自立, 非自立, etc.)
            "pos3": word.feature.pos3,          # Sub-sub-POS
            "pos4": word.feature.pos4,          # Sub-sub-sub-POS
            "cType": word.feature.cType,        # Conjugation type (五段, 一段, etc.)
            "cForm": word.feature.cForm,        # Conjugation form (連用形, 終止形, etc.)
            "reading": word.feature.kana,       # Reading in katakana
        }
        tokens.append(token)
    return tokens
```

### Conjugation Form Explanations (for the tooltip)

When a user clicks a conjugated verb, the tooltip should explain the conjugation. Map MeCab's `cForm` values to human-readable English:

```python
CFORM_EXPLANATIONS = {
    "基本形": "dictionary form (base)",
    "連用形": "conjunctive form (used to chain with other verbs/auxiliary)",
    "連用タ接続": "conjunctive form (connecting to た/だ — past tense)",
    "未然形": "irrealis form (used with ない for negation, or れる for passive)",
    "未然ウ接続": "volitional form (let's ~, shall we ~)",
    "仮定形": "conditional form (if ~)",
    "命令形": "imperative form (command)",
    "仮定縮約": "contracted conditional",
    "体言接続": "noun-modifying form",
    "ガル接続": "form connecting to がる (to show signs of ~)",
    "終止形": "sentence-ending form",
}

CTYPE_EXPLANATIONS = {
    "五段・ラ行": "Godan verb, ら-row (う-verb)",
    "五段・カ行イ音便": "Godan verb, か-row with イ euphonic change",
    "五段・サ行": "Godan verb, さ-row",
    "一段": "Ichidan verb (る-verb)",
    "サ変・スル": "する-verb (suru irregular)",
    "カ変・クル": "くる-verb (kuru irregular)",
    "形容詞・アウオ段": "い-adjective",
    "形容詞・イ段": "い-adjective (イ-row)",
    "特殊・ダ": "だ copula",
    "特殊・デス": "です copula (polite)",
    "特殊・タ": "た auxiliary (past tense)",
    "特殊・ナイ": "ない auxiliary (negation)",
}
```

---

## 6. Dictionary Lookup: jamdict (JMDict)

### Library: jamdict + jamdict-data

**jamdict** provides offline access to JMDict (214,000+ Japanese-English entries), KanjiDic2, and JMnedict, all stored in a local SQLite database. It is the best option for fully offline English definitions.

**Install:**
```bash
pip install jamdict jamdict-data
```

`jamdict-data` includes the pre-compiled SQLite database (~50 MB download). No additional setup required.

### Lookup Function

```python
from jamdict import Jamdict

jam = Jamdict()

def lookup_word(lemma: str) -> dict:
    """
    Look up a word's English definition from JMDict.
    Returns a dict with definitions, POS tags, and readings.
    """
    result = jam.lookup(lemma)

    if not result.entries:
        # Try katakana/hiragana variations
        result = jam.lookup(f"%{lemma}%")

    if not result.entries:
        return {"word": lemma, "definitions": ["(No definition found)"], "pos": []}

    entry = result.entries[0]  # Take the first (most relevant) match

    definitions = []
    pos_tags = []
    for sense in entry.senses:
        # sense.gloss is a list of Gloss objects
        glosses = [str(g) for g in sense.gloss if g.lang == "eng" or g.lang is None]
        if glosses:
            definitions.append("; ".join(glosses))
        pos_tags.extend([str(p) for p in sense.pos])

    readings = [str(k) for k in entry.kana_forms]
    kanji_forms = [str(k) for k in entry.kanji_forms]

    return {
        "word": lemma,
        "kanji_forms": kanji_forms,
        "readings": readings,
        "definitions": definitions,
        "pos": list(set(pos_tags)),
    }
```

### Caching Lookups

Since the same words will be looked up repeatedly (especially common words), cache lookups:

```python
@st.cache_data
def cached_lookup(lemma: str) -> dict:
    return lookup_word(lemma)
```

---

## 7. Clickable Word UI — Story Display

This is the most interactive part of the UI. Each word in the story should be clickable, revealing a tooltip or expandable panel with its translation and grammatical information.

### Implementation Strategy: HTML/CSS Tooltips in st.markdown

Streamlit's native components don't support inline clickable words well. The best approach is to render the story as custom HTML with CSS hover/click tooltips, injected via `st.markdown(html, unsafe_allow_html=True)`.

```python
def render_story_with_tooltips(tokens: list[dict], lookups: dict[str, dict]) -> str:
    """
    Build an HTML string where each word is a clickable span with a tooltip.
    tokens: output of tokenize_story()
    lookups: {lemma: lookup_word(lemma)} pre-fetched for all tokens
    """
    html_parts = []

    for token in tokens:
        surface = token["surface"]
        lemma = token["lemma"] or surface

        # Skip punctuation — render as plain text
        if token["pos"] in ("記号", "補助記号") or surface in "。、！？「」『』（）…―":
            html_parts.append(f'<span class="punct">{surface}</span>')
            continue

        lookup = lookups.get(lemma, {})
        definitions = lookup.get("definitions", ["(unknown)"])
        reading = token.get("reading", "")

        # Build tooltip content
        tooltip_lines = []
        if reading:
            tooltip_lines.append(f"<b>{surface}</b> ({reading})")
        else:
            tooltip_lines.append(f"<b>{surface}</b>")

        if lemma != surface:
            tooltip_lines.append(f"Dictionary form: {lemma}")

        # Conjugation info
        if token.get("cForm") and token["cForm"] != "*":
            form_eng = CFORM_EXPLANATIONS.get(token["cForm"], token["cForm"])
            tooltip_lines.append(f"Form: {form_eng}")

        if token.get("cType") and token["cType"] != "*":
            type_eng = CTYPE_EXPLANATIONS.get(token["cType"], token["cType"])
            tooltip_lines.append(f"Type: {type_eng}")

        # Definitions
        for i, d in enumerate(definitions[:3]):  # Max 3 definitions
            tooltip_lines.append(f"{i+1}. {d}")

        tooltip_html = "<br>".join(tooltip_lines)

        html_parts.append(
            f'<span class="word" tabindex="0">{surface}'
            f'<span class="tooltip">{tooltip_html}</span></span>'
        )

    return "".join(html_parts)
```

### CSS for the Tooltip System

```python
STORY_CSS = """
<style>
.story-container {
    font-size: 22px;
    line-height: 2.0;
    font-family: "Noto Sans JP", "Hiragino Sans", sans-serif;
    padding: 20px;
    background: #fafafa;
    border-radius: 8px;
    margin: 10px 0;
}
.word {
    position: relative;
    cursor: pointer;
    border-bottom: 1px dotted #ccc;
    transition: background 0.2s;
}
.word:hover, .word:focus {
    background: #e8f4fd;
    border-radius: 3px;
}
.tooltip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: #fff;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.6;
    min-width: 250px;
    max-width: 350px;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    text-align: left;
}
.word:hover .tooltip,
.word:focus .tooltip {
    display: block;
}
.punct {
    /* No styling for punctuation */
}
</style>
"""
```

Render the story:
```python
st.markdown(STORY_CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="story-container">{render_story_with_tooltips(tokens, lookups)}</div>',
    unsafe_allow_html=True
)
```

### Mobile Considerations

CSS `:hover` doesn't work on mobile. The `tabindex="0"` attribute allows `:focus` to work on tap, which handles mobile interactions. Add this to the CSS:

```css
.word:focus .tooltip {
    display: block;
}
.word:focus {
    outline: none;
    background: #e8f4fd;
}
```

---

## 8. Persistent Storage

Streamlit `session_state` is lost when the browser tab closes. The assessment results must be persisted to disk.

### Storage Design: JSON File

```python
import json
from pathlib import Path
from datetime import datetime

PROFILE_PATH = Path("experiment/data/user_profile.json")

def save_profile(profile: dict):
    """Save user assessment results to disk."""
    profile["last_updated"] = datetime.now().isoformat()
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2))

def load_profile() -> dict | None:
    """Load saved profile, or return None if no profile exists."""
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text())
    return None

# Profile schema:
# {
#     "threshold_rank": 8500,
#     "estimated_vocab_size": 8500,
#     "assessment_date": "2026-05-21T14:30:00",
#     "last_updated": "2026-05-21T14:30:00",
#     "assessment_responses": [
#         {"rank": 1000, "word": "学校", "known": true},
#         {"rank": 5000, "word": "概念", "known": false},
#         ...
#     ],
#     "stories_generated": 3
# }
```

### Integration with Streamlit Session State

On app startup:

```python
if "profile" not in st.session_state:
    saved = load_profile()
    if saved:
        st.session_state.profile = saved
        st.session_state.app_state = "main"  # Skip to main screen
    else:
        st.session_state.profile = None
        st.session_state.app_state = "assessment"  # Start quiz
```

---

## 9. Application Architecture

### File Structure

```
experiment/
├── data/
│   ├── NLT1.40_freq_list.xlsx    # Source frequency list (provided)
│   ├── freq_list.parquet          # Auto-generated cache (created on first run)
│   └── user_profile.json          # Persisted assessment results (created on first use)
├── app.py                         # Main Streamlit entry point
├── assessment.py                  # Adaptive quiz logic
├── story_generator.py             # LLM prompt building and Ollama interaction
├── tokenizer.py                   # fugashi wrapper, morphological analysis
├── dictionary.py                  # jamdict lookup wrapper
├── ui_components.py               # Tooltip rendering, CSS, reusable UI pieces
├── data_loader.py                 # Excel/Parquet loading and preprocessing
├── storage.py                     # JSON profile persistence
├── config.py                      # Constants, model names, POS mappings
└── requirements.txt               # Python dependencies
```

### requirements.txt

```
streamlit>=1.30.0
pandas>=2.0.0
openpyxl>=3.1.0
pyarrow>=14.0.0
fugashi>=1.3.0
unidic-lite>=1.0.8
jamdict>=0.1a11
jamdict-data>=1.5
ollama>=0.4.0
```

### config.py

```python
# Model configuration
OLLAMA_MODEL = "qwen3:8b"
FALLBACK_MODEL = "qwen3:4b"

# Assessment parameters
WORDS_PER_ROUND = 5
COARSE_ROUNDS = 6
FINE_ROUNDS = 4
TARGET_HIT_RATE = 0.9
BAND_WIDTH = 500

# POS categories
CONTENT_POS = {"名詞", "動詞-自立", "形容詞", "副詞", "形容動詞", "連体詞"}
SKIP_POS = {"記号", "補助記号"}
FUNCTION_POS = {"助詞", "助動詞"}

# Story generation
STORY_MIN_CHARS = 300
STORY_MAX_CHARS = 600
VOCAB_SAMPLE_SIZE = 300

# File paths
FREQ_LIST_PATH = "experiment/data/NLT1.40_freq_list.xlsx"
PARQUET_CACHE_PATH = "experiment/data/freq_list.parquet"
PROFILE_PATH = "experiment/data/user_profile.json"
```

---

## 10. Application Flow — State Machine

The app has three states, managed via `st.session_state.app_state`:

```
                    ┌──────────────────┐
                    │   APP STARTUP    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Load freq list   │
                    │ Check profile    │
                    │ Check Ollama     │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
            Has profile?              No profile
                 │                       │
        ┌────────▼─────────┐   ┌────────▼─────────┐
        │   MAIN SCREEN    │   │   ASSESSMENT     │
        │                  │   │                  │
        │ - Threshold info │   │ - Adaptive quiz  │
        │ - Generate story │   │ - 7-11 rounds    │
        │ - Read story     │   │ - Saves results  │
        │ - Retake button  │   │                  │
        └────────┬─────────┘   └────────┬─────────┘
                 │                       │
                 │   ┌───────────────────┘
                 │   │ Assessment complete
                 │   │
        ┌────────▼───▼─────┐
        │   STORY VIEW     │
        │                  │
        │ - Rendered story │
        │ - Click words    │
        │ - Generate new   │
        │ - Back to main   │
        └──────────────────┘
```

### Main Entry Point (app.py)

```python
import streamlit as st
from data_loader import load_frequency_list
from storage import load_profile, save_profile
from assessment import run_assessment_ui
from story_generator import generate_story_ui

st.set_page_config(page_title="日本語 Vocab Trainer", layout="wide")

# Initialize
df = load_frequency_list()

if "app_state" not in st.session_state:
    profile = load_profile()
    if profile:
        st.session_state.profile = profile
        st.session_state.app_state = "main"
    else:
        st.session_state.profile = None
        st.session_state.app_state = "assessment"

# Sidebar — always visible
with st.sidebar:
    st.title("📚 日本語 Vocab Trainer")
    if st.session_state.profile:
        rank = st.session_state.profile["threshold_rank"]
        st.metric("Your Vocabulary Level", f"Top {rank:,} words")
        st.caption(f"Assessed: {st.session_state.profile['assessment_date'][:10]}")

    if st.button("🔄 Retake Assessment"):
        st.session_state.app_state = "assessment"
        st.session_state.quiz_state = None  # Reset quiz
        st.rerun()

# Route to the right screen
if st.session_state.app_state == "assessment":
    run_assessment_ui(df)
elif st.session_state.app_state == "main":
    generate_story_ui(df)
elif st.session_state.app_state == "story":
    # Story display is handled within generate_story_ui
    generate_story_ui(df)
```

---

## 11. Assessment UI Detail (assessment.py)

```python
import streamlit as st
import random
from config import WORDS_PER_ROUND, CONTENT_POS, COARSE_ROUNDS, FINE_ROUNDS, BAND_WIDTH
from storage import save_profile
from datetime import datetime

def init_quiz_state(df):
    """Initialize the quiz state machine."""
    return {
        "phase": "coarse",           # "coarse" or "fine"
        "round": 0,
        "rank_low": 1,
        "rank_high": len(df),
        "responses": [],             # All {rank, word, known} responses
        "current_words": [],         # Words being shown this round
        "current_answers": {},       # {word: True/False} for this round
        "done": False,
        "threshold": None,
    }

def sample_words(df, center_rank, n=5, band_width=500):
    """Sample content words around a center rank."""
    low = max(1, center_rank - band_width)
    high = min(len(df), center_rank + band_width)
    band = df[(df["rank"] >= low) & (df["rank"] <= high)]
    content = band[band["pos"].apply(
        lambda p: any(p.startswith(cp) for cp in CONTENT_POS)
    )]
    if len(content) < n:
        content = band[~band["pos"].str.startswith("助")]
    return content.sample(n=min(n, len(content))).to_dict("records")

def run_assessment_ui(df):
    st.header("📝 Vocabulary Assessment")
    st.write("We'll show you words at different frequency levels to estimate your vocabulary size. "
             "For each word, indicate whether you know its meaning.")

    # Initialize quiz state
    if "quiz_state" not in st.session_state or st.session_state.quiz_state is None:
        st.session_state.quiz_state = init_quiz_state(df)

    qs = st.session_state.quiz_state

    if qs["done"]:
        _show_results(qs)
        return

    # Sample new words if needed
    if not qs["current_words"]:
        mid = (qs["rank_low"] + qs["rank_high"]) // 2
        qs["current_words"] = sample_words(df, mid)
        qs["current_answers"] = {}

    # Display progress
    total_rounds = COARSE_ROUNDS + FINE_ROUNDS
    st.progress(qs["round"] / total_rounds)
    st.caption(f"Round {qs['round'] + 1} of ~{total_rounds}")

    # Display word cards
    cols = st.columns(WORDS_PER_ROUND)
    for i, word_data in enumerate(qs["current_words"]):
        with cols[i % WORDS_PER_ROUND]:
            lemma = word_data["lemma"]
            st.markdown(f"### {lemma}")
            if word_data.get("reading"):
                st.caption(word_data["reading"])

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Know", key=f"know_{qs['round']}_{i}"):
                    qs["current_answers"][lemma] = True
            with c2:
                if st.button("❌ Don't", key=f"dont_{qs['round']}_{i}"):
                    qs["current_answers"][lemma] = False

    # Submit round
    if len(qs["current_answers"]) == len(qs["current_words"]):
        if st.button("Next Round →", type="primary"):
            _process_round(qs, df)
            st.rerun()

def _process_round(qs, df):
    """Process the current round's answers and update search bounds."""
    for word_data in qs["current_words"]:
        lemma = word_data["lemma"]
        qs["responses"].append({
            "rank": word_data["rank"],
            "word": lemma,
            "known": qs["current_answers"].get(lemma, False),
        })

    # Calculate hit rate for this round
    answers = list(qs["current_answers"].values())
    hit_rate = sum(answers) / len(answers) if answers else 0

    mid = (qs["rank_low"] + qs["rank_high"]) // 2

    if qs["phase"] == "coarse":
        if hit_rate >= 0.8:
            qs["rank_low"] = mid
        elif hit_rate < 0.5:
            qs["rank_high"] = mid
        else:
            qs["rank_high"] = mid + (qs["rank_high"] - mid) // 3

        qs["round"] += 1

        if qs["rank_high"] - qs["rank_low"] < 2000 or qs["round"] >= COARSE_ROUNDS:
            qs["phase"] = "fine"
    else:
        # Fine phase — just accumulate data
        qs["round"] += 1

        if qs["round"] >= COARSE_ROUNDS + FINE_ROUNDS:
            qs["done"] = True
            qs["threshold"] = _estimate_threshold(qs["responses"])

    # Reset for next round
    qs["current_words"] = []
    qs["current_answers"] = {}

def _estimate_threshold(responses):
    """Estimate the vocabulary threshold from all responses."""
    from collections import defaultdict
    bands = defaultdict(list)
    for r in responses:
        band = (r["rank"] // 500) * 500
        bands[band].append(r["known"])

    for band_start in sorted(bands.keys()):
        hit_rate = sum(bands[band_start]) / len(bands[band_start])
        if hit_rate < 0.7:
            return band_start
    return max(bands.keys()) + 500

def _show_results(qs):
    """Display assessment results and save profile."""
    threshold = qs["threshold"]

    st.success(f"Assessment complete! Your estimated vocabulary covers the top **{threshold:,}** most frequent words.")
    st.balloons()

    # Save to profile
    profile = {
        "threshold_rank": threshold,
        "estimated_vocab_size": threshold,
        "assessment_date": datetime.now().isoformat(),
        "assessment_responses": qs["responses"],
        "stories_generated": 0,
    }
    save_profile(profile)
    st.session_state.profile = profile

    if st.button("Start Reading Stories →", type="primary"):
        st.session_state.app_state = "main"
        st.session_state.quiz_state = None
        st.rerun()
```

---

## 12. Story Generation UI Detail (story_generator.py)

```python
import streamlit as st
import ollama
from config import OLLAMA_MODEL, CONTENT_POS, VOCAB_SAMPLE_SIZE
from tokenizer import tokenize_story
from dictionary import cached_lookup
from ui_components import render_story_with_tooltips, STORY_CSS

def generate_story_ui(df):
    profile = st.session_state.profile
    threshold = profile["threshold_rank"]

    st.header("📖 Story Generator")
    st.write(f"Generating stories using your top {threshold:,} vocabulary words.")

    # Topic selection
    topic = st.text_input("Story topic (optional — leave blank for a random topic):",
                          placeholder="e.g., 猫の冒険、学校生活、旅行...")

    # Generate button
    if st.button("✨ Generate New Story", type="primary"):
        with st.spinner("Writing your story... (this may take 15-30 seconds)"):
            prompt = build_story_prompt(df, threshold, topic if topic else None)
            raw_story = generate_story(prompt)
            story_text = strip_thinking(raw_story)
            st.session_state.current_story = story_text

    # Display story if one exists
    if "current_story" in st.session_state and st.session_state.current_story:
        story = st.session_state.current_story

        # Tokenize
        tokens = tokenize_story(story)

        # Pre-fetch all lookups
        unique_lemmas = set(t["lemma"] for t in tokens
                          if t["lemma"] and t["pos"] not in ("記号", "補助記号"))
        lookups = {lemma: cached_lookup(lemma) for lemma in unique_lemmas}

        # Render with tooltips
        st.markdown(STORY_CSS, unsafe_allow_html=True)
        html = render_story_with_tooltips(tokens, lookups)
        st.markdown(f'<div class="story-container">{html}</div>',
                    unsafe_allow_html=True)

        st.caption("💡 Click or hover on any word to see its translation and grammar info.")

        # Story stats
        with st.expander("Story Statistics"):
            content_tokens = [t for t in tokens if t["pos"] not in ("記号", "補助記号")]
            st.write(f"Total words: {len(content_tokens)}")
            st.write(f"Unique words: {len(unique_lemmas)}")

            # Check how many words are within threshold
            known_vocab_set = set(df[df["rank"] <= threshold]["lemma"])
            in_vocab = sum(1 for t in content_tokens
                         if t["lemma"] in known_vocab_set)
            coverage = in_vocab / len(content_tokens) * 100 if content_tokens else 0
            st.write(f"Words within your vocabulary: {coverage:.1f}%")
```

---

## 13. Potential Issues & Mitigations

### Issue 1: LLM Generates Words Outside the Vocabulary Threshold

**Problem:** Even with careful prompting, LLMs don't strictly obey vocabulary constraints. The model may use words outside the user's known range.

**Mitigation (Post-Generation Validation):**

```python
def validate_story_vocab(story_tokens, known_vocab_set):
    """
    Check each content word against the user's vocabulary.
    Returns a list of out-of-vocabulary words.
    """
    oov_words = []
    for token in story_tokens:
        if token["pos"] in ("記号", "補助記号", "助詞", "助動詞"):
            continue  # Grammar words are always OK
        if token["lemma"] not in known_vocab_set:
            oov_words.append(token)
    return oov_words
```

If more than 15% of content words are out of vocabulary, regenerate with a stricter prompt. This is expected to happen occasionally — the tooltip system ensures the user can still look up any unknown word, so it degrades gracefully rather than breaking.

### Issue 2: Large Excel File Loading Time

**Problem:** If the file has millions of rows, loading from `.xlsx` can take 30+ seconds.

**Mitigation:** The Parquet cache strategy described in Section 2. On first run, convert to Parquet. Subsequent loads take <1 second.

### Issue 3: Ollama Not Installed or Model Not Pulled

**Problem:** The app crashes if Ollama isn't running.

**Mitigation:** Check at startup and show friendly instructions. Never call `ollama.chat()` without first verifying connectivity:

```python
def check_ollama_ready(model: str = OLLAMA_MODEL) -> tuple[bool, str]:
    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        if any(model in m for m in available):
            return True, ""
        return False, f"Model '{model}' not found. Run: ollama pull {model}"
    except Exception as e:
        return False, f"Cannot connect to Ollama: {e}. Is it running?"
```

### Issue 4: fugashi/MeCab Segmentation Mismatches with Frequency List

**Problem:** MeCab may tokenize a word differently from how it appears in the frequency list. For example, MeCab might produce lemma "為る" while the frequency list has "する".

**Mitigation:** Build a lookup set from the frequency list's lemma column and also index by reading (katakana). When checking if a token is "known," check both the lemma and reading:

```python
known_lemmas = set(df[df["rank"] <= threshold]["lemma"])
known_readings = set(df[df["rank"] <= threshold]["reading"].dropna())

def is_word_known(token):
    return (token["lemma"] in known_lemmas or
            token.get("reading") in known_readings or
            token["surface"] in known_lemmas)
```

### Issue 5: Session State Loss on Browser Refresh

**Problem:** Streamlit session state is lost on tab close/refresh.

**Mitigation:** This is already handled by the JSON persistence layer. Assessment results are saved to disk immediately upon completion. The in-progress quiz state is only in session_state (acceptable — restarting a 3-minute quiz is not a significant burden). The story is regenerated on each request (also acceptable).

### Issue 6: jamdict Lookup Failures for Conjugated or Compound Words

**Problem:** jamdict may not find entries for some surface forms or unusual compound words.

**Mitigation:** The lookup function already tries wildcard matching as a fallback. Additionally, always look up the lemma (dictionary form) rather than the surface form, since fugashi provides the lemma.

---

## 14. Setup Instructions (for the end user)

Include a `README.md` in the project with these setup steps:

```markdown
# Japanese Vocabulary Trainer — Setup

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from https://ollama.com

## Installation

# Clone/copy the project
cd experiment

# Install Python dependencies
pip install -r requirements.txt

# Pull the LLM model (one-time, ~4.7 GB download)
ollama pull qwen3:8b

# If you have limited hardware (< 6GB VRAM/RAM), use the smaller model:
# ollama pull qwen3:4b
# Then edit config.py: OLLAMA_MODEL = "qwen3:4b"

## Running

# Make sure Ollama is running (it usually starts automatically)
ollama serve  # (if not auto-started)

# Launch the app
streamlit run app.py

## First Use

The app will open in your browser. On first launch, you'll take a vocabulary
assessment (~3 minutes). After that, you can generate stories tailored to your level.
```

---

## 15. Dependency Version Matrix

| Package | Min Version | Purpose | Size |
|---------|-------------|---------|------|
| streamlit | 1.30.0 | UI framework | ~50 MB |
| pandas | 2.0.0 | Data manipulation | ~40 MB |
| openpyxl | 3.1.0 | Excel file reading | ~8 MB |
| pyarrow | 14.0.0 | Parquet read/write | ~80 MB |
| fugashi | 1.3.0 | MeCab wrapper (tokenizer) | ~2 MB |
| unidic-lite | 1.0.8 | MeCab dictionary | ~47 MB |
| jamdict | 0.1a11 | JMDict dictionary interface | ~1 MB |
| jamdict-data | 1.5 | JMDict SQLite database | ~49 MB |
| ollama | 0.4.0 | Ollama Python client | ~1 MB |

**Total Python deps:** ~278 MB
**Ollama model (qwen3:8b Q4):** ~4.7 GB

---

## 16. Summary of Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Qwen3 8B via Ollama | Best Japanese quality per VRAM; runs on 6GB GPU or CPU; Apache 2.0 |
| Tokenizer | fugashi + unidic-lite | Industry standard MeCab wrapper; fast; provides conjugation info |
| Dictionary | jamdict (JMDict) | 214K entries; fully offline; SQLite-backed; includes POS and readings |
| Assessment algo | Batched adaptive binary search | Noise-tolerant; completes in 35–55 words; no IRT calibration needed |
| Persistence | JSON file on disk | Simple; no database needed; survives restarts; human-readable |
| Word tooltips | HTML/CSS in st.markdown | Works on hover (desktop) and tap (mobile); no extra JS dependencies |
| Data caching | Parquet auto-cache | 10–50x faster than re-reading Excel on every launch |
| Thinking mode | Strip post-generation | More reliable than prompt-level control across Ollama versions |
