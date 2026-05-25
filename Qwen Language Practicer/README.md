# 日本語 Vocab Trainer

> An AI-powered Japanese reading tool that meets you exactly where your vocabulary is.

---

## The Story Behind This

I've been studying Japanese on and off for years, and the thing that always frustrated me about conventional tools was the mismatch. Anki decks are either too easy or too hard. Graded readers top out fast. Real native content — manga, novels, news — is a wall of unknown words that destroys reading flow and makes it feel hopeless.

The study method I actually wanted was simple in concept but hard to execute by hand: **read real sentences where I know almost every word, and have instant access to anything I don't.** That's how children acquire language. That's how extensive reading works in theory. The problem is that building personalized content at exactly your level requires knowing your level precisely, and generating content calibrated to a specific vocabulary range is essentially impossible without a capable language model.

When Qwen3 dropped — a genuinely powerful model that runs locally on consumer hardware and handles Japanese with impressive fluency — I realized I finally had all the pieces. Qwen3 runs through Ollama, which means no API keys, no subscriptions, no data leaving my machine. The NLT frequency list gives me a ranked vocabulary of tens of thousands of Japanese words. MeCab handles morphological analysis so I can tokenize any text the model produces. JMDict provides offline dictionary definitions. Streamlit ties it all into a usable interface in a weekend.

The result is a tool that does something I genuinely couldn't get anywhere else: it figures out exactly how many Japanese words I know, then generates fresh short stories using only those words. Every word in every story is interactive — hover it and you get the reading, dictionary form, conjugation type, and English definition. No switching tabs, no copy-pasting into Jisho, no interrupting the reading flow.

I built this for myself. It reflects the way I think about language acquisition: figure out what you know, stay in that zone, read a lot, look things up frictionlessly. If it's useful to you too, great.

---

## What It Does

**1. Adaptive Vocabulary Assessment**

When you first launch the app, you take a ~3-minute quiz. It shows you Japanese words one at a time (flashcard style), you reveal the answer, and you honestly mark whether you knew it. Under the hood, it runs a binary search across a frequency-ranked vocabulary list: if you know words at rank 3,000 it tests higher; if you don't know them it tests lower. It narrows its estimate through coarse and fine phases until it can confidently pin your vocabulary size to a range like "you know approximately 2,400–2,800 of the most common Japanese words."

The result is saved locally. You never have to retake it unless you want to.

**2. Calibrated Story Generation**

With your vocabulary level known, the story generator samples from the words you actually know, hands them to Qwen3 as a constraint, and asks it to write a short, complete story (about 10 sentences) using only vocabulary at or below your level. The model uses its thinking mode internally to reason through the constraint before writing, so the output tends to stay within bounds.

You can specify a topic — 猫の冒険, 学校生活, 旅行, anything — or leave it blank for a random one.

**3. Interactive Reading**

Every word in the generated story is a hoverable tooltip. Click or hover any word to see:
- Its reading (furigana)
- The dictionary form (if it's conjugated)
- The conjugation type and form in English (e.g., "Godan verb, ら-row" / "conditional form")
- Up to three English definitions from JMDict

No internet required. All lookups are offline.

**4. Story Statistics**

After each story, an expandable panel shows total token count, unique word count, and what percentage of the story falls within your vocabulary threshold. Words outside your threshold are flagged but still fully clickable.

---

## How It Works (Technical)

| Component | Role |
|---|---|
| **Qwen3:8b / 4b** via Ollama | Story generation — runs locally, no API key |
| **NLT 1.40 frequency list** | ~50,000 Japanese lemmas ranked by corpus frequency |
| **fugashi + MeCab** | Morphological analysis — tokenizes Japanese text into lemmas, readings, POS tags, conjugation info |
| **jamdict (JMDict)** | Offline Japanese–English dictionary lookups |
| **Streamlit** | Web UI — runs in browser, no frontend build step |

The assessment uses a two-phase binary search. The coarse phase narrows the search window from [1, 12,000] down to a ~2,000-word range in 6 rounds. The fine phase then tightens within that window. Threshold estimation blends the binary search bracket midpoint with the centroid of known/unknown responses to produce a refined estimate with a confidence interval.

Story prompts are built by sampling ~300 content words (nouns, verbs, adjectives, adverbs) from your known vocabulary and passing them to the model as examples of the allowed range. Function words (particles, auxiliaries) are excluded from the constraint since the model handles those independently.

The model selector in the sidebar reads your available GPU VRAM via nvidia-smi and recommends the appropriate model size automatically: 8b if you have ~5 GB free, 4b if you're tighter.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- MeCab (installed automatically via `unidic-lite` — no manual setup needed on Windows)

---

## Installation

```bash
# Clone or download this folder, then:
pip install -r requirements.txt

# Pull the LLM (one-time download — pick one)
ollama pull qwen3:8b    # ~5 GB, better quality, needs ~5.2 GB VRAM
ollama pull qwen3:4b    # ~3 GB, faster, needs ~2.8 GB VRAM

# If using 4b, set in config.py: OLLAMA_MODEL = "qwen3:4b"
```

---

## Running

```bash
# Make sure Ollama is running (usually auto-starts after install)
# If not: ollama serve

streamlit run app.py
```

The app opens in your browser automatically at `http://localhost:8501`.

---

## First Use

On first launch you'll take the vocabulary assessment (35–55 words, ~3 minutes). Be honest with yourself — if you had to guess, mark it wrong. The assessment only works if your answers reflect genuine recall.

Results are saved to `data/user_profile.json` and persist across restarts. You can retake the assessment anytime from the sidebar button.

After the assessment, generate your first story. Try a topic you actually care about — engagement matters more than the specific words.

---

## Hardware Notes

The app auto-detects your GPU and recommends a model. General guidance:

| VRAM | Recommendation |
|---|---|
| 6 GB+ | qwen3:8b — better story quality and constraint adherence |
| 4–6 GB | qwen3:8b may partially spill to CPU (slower but works) |
| Under 4 GB | qwen3:4b — fits fully on GPU, good results |
| CPU only | qwen3:4b — slow but functional |

---

## File Structure

```
├── app.py               — Streamlit entry point and routing
├── assessment.py        — Adaptive quiz logic and binary search
├── story_generator.py   — Prompt construction and Ollama interaction
├── tokenizer.py         — fugashi/MeCab morphological analysis
├── dictionary.py        — jamdict offline lookups with caching
├── ui_components.py     — Tooltip HTML/CSS rendering
├── data_loader.py       — Excel → Parquet frequency list loader
├── storage.py           — User profile persistence
├── config.py            — Model names, paths, tuning constants
├── data/
│   ├── NLT1.40_freq_list.xlsx   — Source frequency list
│   ├── freq_list.parquet        — Auto-generated cache (faster loads)
│   └── user_profile.json        — Your assessment results (auto-created)
└── requirements.txt
```

---

## Acknowledgments

- [NLT (Nihongo no Tango) frequency list](http://www17408ui.sakura.ne.jp/tatsum/database.html) by Tono, Yamazaki & Maekawa — the vocabulary backbone
- [Qwen3](https://qwenlm.github.io/) by Alibaba Cloud — the model that made this actually work
- [Ollama](https://ollama.com) — local model serving without the ops overhead
- [JMDict](https://www.edrdg.org/jmdict/j_jmdict.html) / [jamdict](https://github.com/neocl/jamdict) — the offline dictionary
- [fugashi](https://github.com/polm/fugashi) + [unidic-lite](https://github.com/polm/unidic-lite) — Japanese tokenization that just works on Windows
