import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

from src.utils import get_global_store

REQUIRED_COLUMNS = [
    "Participant",
    "Scoring metric",
    "Recall",
    "Accuracy",
    "Hospitalized",
    "Edible but uneaten",
    "submission_time",
    "batch",
]


def _open_spreadsheet():
    """Opens the raw gspread client to allow structural changes (like adding tabs).
    Requires 'connections.gsheets' in your secrets.
    """
    creds_dict = dict(st.secrets["connections"]["gsheets"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(creds_dict["spreadsheet"])


def ensure_batch_sheet_exists(batch: str) -> None:
    """Checks if a worksheet exists for the batch; creates and initializes it if not."""
    store = get_global_store()
    conn = store["gsheet_conn"]

    try:
        # Try to read to check existence
        conn.read(worksheet=batch, ttl=0)
    except WorksheetNotFound:
        # If not found, use gspread to add the tab
        sh = _open_spreadsheet()
        sh.add_worksheet(title=batch, rows="1000", cols="10")

        # Initialize the new tab with the correct headers
        headers = [
            "Participant",
            "Scoring metric",
            "Recall",
            "Accuracy",
            "Hospitalized",
            "Edible but uneaten",
            "submission_time",
            "batch",
        ]
        empty_df = pd.DataFrame(columns=headers)
        conn.update(worksheet=batch, data=empty_df)
        st.toast(f"Created new database tab for batch: {batch}")
    except Exception as e:
        st.error(f"Failed to verify/create worksheet: {e}")
