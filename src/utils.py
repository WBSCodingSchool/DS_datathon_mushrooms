from typing import Any

import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

RESULTS_PATH = "data/true_y.csv"


@st.cache_resource
def get_global_store() -> dict[str, Any]:
    """Initialize the global in-memory store to minimize GSheet calls."""
    return {
        "submissions": {},
        "gsheet_conn": None,
        "batches": None,
        "batches_last_updated": None,
    }


def state_inits() -> None:
    """Initialize session state and handles GSheet connection logic."""
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "batch" not in st.session_state:
        st.session_state.batch = None
    if "alltime" not in st.session_state:
        st.session_state.alltime = False

    store = get_global_store()

    default_keys = {
        "submissions": {},
        "batches": None,
        "batches_last_updated": None,
        "gsheet_conn": None,
        "configured_batches": set(),
    }

    for key, default_value in default_keys.items():
        if key not in store:
            store[key] = default_value

    if store.get("gsheet_conn") is None:
        store["gsheet_conn"] = st.connection("gsheets", type=GSheetsConnection)

    if st.session_state.alltime and store.get("alltime_submissions") is None:
        load_alltime_data(store)


def load_alltime_data(store) -> None:
    try:
        batches_df = store["gsheet_conn"].read(worksheet="Batches", ttl=0)
        batches = batches_df["Batch"].tolist()

        worksheet_titles = [b for b in batches if b not in ["Batches", "anonymous"]]

        dfs = []
        for ws_name in worksheet_titles:
            try:
                df = store["gsheet_conn"].read(worksheet=ws_name, ttl=0)
                if df is not None and not df.empty:
                    dfs.append(df)
            except Exception:
                continue

        if dfs:
            store["alltime_submissions"] = pd.concat(dfs, ignore_index=True)
        else:
            store["alltime_submissions"] = pd.DataFrame()

    except Exception as e:
        st.error(f"Error aggregating global data: {e}")
        store["alltime_submissions"] = pd.DataFrame()
