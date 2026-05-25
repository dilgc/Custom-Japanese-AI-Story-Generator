import os
import pandas as pd
import streamlit as st
from config import FREQ_LIST_PATH, PARQUET_CACHE_PATH


@st.cache_data(show_spinner="Loading vocabulary list (first run may take a minute)...")
def load_frequency_list() -> pd.DataFrame:
    if os.path.exists(PARQUET_CACHE_PATH):
        return pd.read_parquet(PARQUET_CACHE_PATH)

    df = pd.read_excel(
        FREQ_LIST_PATH,
        engine="openpyxl",
        sheet_name="NLT 1.40頻度リスト",
        header=0,
    )
    df.columns = ["lemma", "pos", "reading", "freq"]

    # Strip whitespace from string columns
    for col in ["lemma", "pos", "reading"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace("nan", "")

    # Filter out punctuation/symbols and category headers
    df = df[df["pos"] != "記号"]
    df = df[~df["lemma"].str.startswith("【", na=False)]

    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    os.makedirs(os.path.dirname(PARQUET_CACHE_PATH), exist_ok=True)
    df.to_parquet(PARQUET_CACHE_PATH, index=False)

    return df
