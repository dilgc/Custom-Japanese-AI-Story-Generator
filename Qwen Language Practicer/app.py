import streamlit as st

from assessment import run_assessment_ui
from data_loader import load_frequency_list
from storage import load_profile
from story_generator import generate_story_ui

st.set_page_config(
    page_title="日本語 Vocab Trainer",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

df = load_frequency_list()

if "app_state" not in st.session_state:
    profile = load_profile()
    if profile:
        st.session_state.profile = profile
        st.session_state.app_state = "main"
    else:
        st.session_state.profile = None
        st.session_state.app_state = "assessment"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 日本語 Vocab Trainer")
    st.caption(f"Vocabulary list: {len(df):,} entries")

    if st.session_state.profile:
        p = st.session_state.profile
        rank = p["threshold_rank"]
        st.metric("Your vocabulary level", f"~{rank:,} words")
        ci_low  = p.get("ci_low")
        ci_high = p.get("ci_high")
        if ci_low and ci_high and ci_high != ci_low:
            st.caption(f"Range: {ci_low:,} – {ci_high:,}")
        date_str = p.get("assessment_date", "")[:10]
        if date_str:
            st.caption(f"Assessed: {date_str}")

    st.divider()

    if st.button("🔄 Retake Assessment"):
        st.session_state.app_state = "assessment"
        st.session_state.quiz_state = None
        st.rerun()

    if st.session_state.profile and st.session_state.app_state != "main":
        if st.button("🏠 Back to Story Generator"):
            st.session_state.app_state = "main"
            st.rerun()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

if st.session_state.app_state == "assessment":
    run_assessment_ui(df)

elif st.session_state.app_state in ("main", "story"):
    # Update story count when a story exists
    if (
        "current_story" in st.session_state
        and st.session_state.current_story
        and st.session_state.profile
    ):
        if "stories_generated" not in st.session_state:
            st.session_state.stories_generated = 0

    generate_story_ui(df)
