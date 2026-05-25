from collections import defaultdict
from datetime import datetime

import streamlit as st

from config import BAND_WIDTH, COARSE_ROUNDS, CONTENT_POS, FINE_ROUNDS, WORDS_PER_ROUND
from dictionary import cached_lookup
from storage import save_profile


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

INITIAL_HIGH = 12_000   # Starting upper bound — roughly JLPT N2 territory.
                        # Expanded automatically if the user aces the top end.
MAX_HIGH = 50_000       # Hard ceiling (above this are extremely rare lemmas).


def init_quiz_state(df) -> dict:
    return {
        "phase": "coarse",
        "round": 0,
        "rank_low": 1,
        "rank_high": min(INITIAL_HIGH, len(df)),
        "vocab_size": len(df),
        "responses": [],
        "current_words": [],
        "current_answers": {},  # lemma -> True/False
        "flipped": set(),       # lemmas whose card has been revealed
        "done": False,
        "threshold": None,
    }


_KATAKANA_RANGE = frozenset(
    chr(c) for c in range(ord("ァ"), ord("ン") + 1)
) | {"ー", "・", "ヴ"}


def _is_katakana_word(lemma: str) -> bool:
    """True if the word is written entirely in katakana (loanword)."""
    return bool(lemma) and all(c in _KATAKANA_RANGE for c in lemma)


def _sample_words(df, center_rank: int, n: int = WORDS_PER_ROUND,
                  band_width: int = BAND_WIDTH) -> list[dict]:
    low = max(1, center_rank - band_width)
    high = min(len(df), center_rank + band_width)
    band = df[(df["rank"] >= low) & (df["rank"] <= high)]

    # Content words only, excluding all-katakana loanwords
    content = band[
        band["pos"].apply(lambda p: any(p.startswith(cp) for cp in CONTENT_POS))
        & ~band["lemma"].apply(_is_katakana_word)
    ]
    if len(content) < n:
        # Fallback: anything except particles, still no katakana-only words
        content = band[
            ~band["pos"].str.startswith("助")
            & ~band["lemma"].apply(_is_katakana_word)
        ]
    if len(content) == 0:
        content = band  # last resort: use whatever is there

    return content.sample(n=min(n, len(content))).to_dict("records")


def _estimate_threshold(responses: list[dict], rank_low: int, rank_high: int) -> int:
    """
    Estimate the vocabulary threshold precisely.

    The binary search already bracketed the threshold to [rank_low, rank_high].
    The midpoint of that bracket is our primary estimate.  We then refine it
    by looking at the actual known/unknown responses that fell inside the final
    window: the threshold should sit between the average rank of words the user
    knew and the average rank of words they didn't know in that window.
    """
    primary = (rank_low + rank_high) // 2

    # Responses that landed inside the final search window
    local = [r for r in responses if rank_low <= r["rank"] <= rank_high]

    if len(local) < 2:
        return primary

    known_ranks   = [r["rank"] for r in local if r["known"]]
    unknown_ranks = [r["rank"] for r in local if not r["known"]]

    if known_ranks and unknown_ranks:
        # Threshold sits between the centroid of known words and the centroid
        # of unknown words inside the window — blend with the bracket midpoint.
        avg_known   = sum(known_ranks)   / len(known_ranks)
        avg_unknown = sum(unknown_ranks) / len(unknown_ranks)
        secondary = int((avg_known + avg_unknown) / 2)
        return int((primary + secondary) / 2)

    if not unknown_ranks:
        # All words in the window were known → threshold is near the top
        return int((primary + rank_high) / 2)

    # All unknown → threshold is near the bottom
    return int((primary + rank_low) / 2)


def _process_round(qs: dict, df) -> None:
    for word_data in qs["current_words"]:
        lemma = word_data["lemma"]
        qs["responses"].append({
            "rank": word_data["rank"],
            "word": lemma,
            "known": qs["current_answers"].get(lemma, False),
        })

    answers = list(qs["current_answers"].values())
    hit_rate = sum(answers) / len(answers) if answers else 0
    mid = (qs["rank_low"] + qs["rank_high"]) // 2

    # --- Binary search bounds update (same logic for both phases) -----------
    if hit_rate >= 0.8:
        # User knows words at this level — threshold is higher, expand low end.
        # If we're already near the ceiling, push the ceiling up.
        if mid >= qs["rank_high"] - 500 and qs["rank_high"] < MAX_HIGH:
            new_high = min(qs["rank_high"] * 2, MAX_HIGH, qs["vocab_size"])
            qs["rank_high"] = new_high
        else:
            qs["rank_low"] = mid
    elif hit_rate < 0.5:
        qs["rank_high"] = mid
    else:
        # Ambiguous — narrow the upper half more gently
        qs["rank_high"] = mid + (qs["rank_high"] - mid) // 2

    qs["round"] += 1

    # --- Phase transitions --------------------------------------------------
    window = qs["rank_high"] - qs["rank_low"]
    confident = window <= 500          # tight enough to stop early
    coarse_done = window <= 2_000 or qs["round"] >= COARSE_ROUNDS
    max_rounds_hit = qs["round"] >= 10  # hard cap regardless of phase

    if qs["phase"] == "coarse":
        if coarse_done:
            qs["phase"] = "fine"

    if confident or max_rounds_hit:
        qs["done"] = True
        qs["threshold"] = _estimate_threshold(
            qs["responses"], qs["rank_low"], qs["rank_high"]
        )
        qs["ci_low"]  = qs["rank_low"]
        qs["ci_high"] = qs["rank_high"]

    qs["current_words"] = []
    qs["current_answers"] = {}
    qs["flipped"] = set()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_CARD_CSS = """
<style>
.fc-word {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    font-family: "Noto Sans JP", "Yu Gothic", "Hiragino Sans", sans-serif;
    padding: 12px 0 8px 0;
    letter-spacing: 2px;
}
.fc-reading {
    font-size: 18px;
    text-align: center;
    color: #94a3b8;
    margin-bottom: 4px;
    font-family: "Noto Sans JP", sans-serif;
}
.fc-def {
    font-size: 15px;
    color: #e2e8f0;
    line-height: 1.6;
    margin-top: 6px;
}
.fc-pos {
    font-size: 12px;
    color: #64748b;
    margin-top: 4px;
}
.fc-rank {
    font-size: 11px;
    color: #475569;
    text-align: right;
}
</style>
"""


def run_assessment_ui(df) -> None:
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    st.header("📝 Vocabulary Assessment")
    st.write(
        "Think about what each word means, then reveal to check yourself. "
        "Mark it correct only if you genuinely knew the meaning."
    )

    if "quiz_state" not in st.session_state or st.session_state.quiz_state is None:
        st.session_state.quiz_state = init_quiz_state(df)

    qs = st.session_state.quiz_state

    if qs["done"]:
        _show_results(qs)
        return

    # Sample words for this round if needed
    if not qs["current_words"]:
        mid = (qs["rank_low"] + qs["rank_high"]) // 2
        qs["current_words"] = _sample_words(df, mid)
        qs["current_answers"] = {}
        qs["flipped"] = set()

    total_rounds = COARSE_ROUNDS + FINE_ROUNDS
    st.progress(qs["round"] / total_rounds)
    lo, hi = qs["rank_low"], qs["rank_high"]
    mid = (lo + hi) // 2
    st.caption(
        f"Round {qs['round'] + 1} of ~{total_rounds}  ·  "
        f"Testing around rank {mid:,}  ·  "
        f"Search window: {lo:,} – {hi:,}  ·  "
        f"Answered {len(qs['current_answers'])}/{len(qs['current_words'])}"
    )

    st.divider()

    for i, word_data in enumerate(qs["current_words"]):
        lemma = word_data["lemma"]
        reading = word_data.get("reading", "") or ""
        rank = word_data.get("rank", "?")
        is_flipped = lemma in qs["flipped"]
        is_answered = lemma in qs["current_answers"]

        with st.container(border=True):
            # Always show the word
            st.markdown(f'<div class="fc-word">{lemma}</div>', unsafe_allow_html=True)

            if not is_flipped:
                # Face-down: just the word and a reveal button
                _, btn_col, _ = st.columns([2, 2, 2])
                with btn_col:
                    if st.button("Reveal →", key=f"flip_{qs['round']}_{i}",
                                 use_container_width=True):
                        qs["flipped"].add(lemma)
                        st.rerun()

            else:
                # Face-up: show reading, definition, then judgment buttons
                lookup = cached_lookup(lemma)

                # Prefer JMDict reading (canonical dictionary form) over the
                # frequency-list reading, which reflects the corpus inflection
                # (e.g. イタク for 痛い instead of イタイ).
                jmdict_readings = lookup.get("readings", [])
                display_reading = jmdict_readings[0] if jmdict_readings else reading

                if display_reading:
                    st.markdown(f'<div class="fc-reading">{display_reading}</div>',
                                unsafe_allow_html=True)

                defs = lookup.get("definitions", [])
                pos_tags = lookup.get("pos", [])

                if defs:
                    numbered = "<br>".join(
                        f"{j+1}. {d}" for j, d in enumerate(defs[:4])
                    )
                    st.markdown(f'<div class="fc-def">{numbered}</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fc-def"><em>No definition found</em></div>',
                                unsafe_allow_html=True)

                if pos_tags:
                    st.markdown(
                        f'<div class="fc-pos">{" · ".join(pos_tags[:3])}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(f'<div class="fc-rank">Frequency rank #{rank:,}</div>',
                            unsafe_allow_html=True)

                st.write("")  # spacer

                if not is_answered:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅  I knew it", key=f"know_{qs['round']}_{i}",
                                     use_container_width=True, type="primary"):
                            qs["current_answers"][lemma] = True
                            st.rerun()
                    with c2:
                        if st.button("❌  I didn't know it", key=f"dont_{qs['round']}_{i}",
                                     use_container_width=True):
                            qs["current_answers"][lemma] = False
                            st.rerun()
                else:
                    result = qs["current_answers"][lemma]
                    if result:
                        st.success("Marked: knew it ✅", icon=None)
                    else:
                        st.error("Marked: didn't know it ❌", icon=None)

        # Small gap between cards
        st.write("")

    st.divider()

    all_answered = len(qs["current_answers"]) == len(qs["current_words"])
    all_revealed = all(w["lemma"] in qs["flipped"] for w in qs["current_words"])

    if not all_revealed:
        st.caption("Reveal all cards before moving on.")
    elif not all_answered:
        st.caption("Mark every card before moving on.")
    else:
        if st.button("Next Round →", type="primary"):
            _process_round(qs, df)
            st.rerun()


def _show_results(qs: dict) -> None:
    threshold = qs["threshold"]
    ci_low  = qs.get("ci_low",  threshold)
    ci_high = qs.get("ci_high", threshold)
    window  = ci_high - ci_low

    if window <= 500:
        confidence_label = "high confidence"
    elif window <= 1500:
        confidence_label = "moderate confidence"
    else:
        confidence_label = "low confidence — consider retaking for a tighter estimate"

    st.success(
        f"Assessment complete!  \n"
        f"Estimated vocabulary: **{threshold:,}** most-frequent words  \n"
        f"Confidence range: **{ci_low:,} – {ci_high:,}** ({confidence_label})"
    )
    st.balloons()

    profile = {
        "threshold_rank": threshold,
        "estimated_vocab_size": threshold,
        "ci_low": ci_low,
        "ci_high": ci_high,
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
