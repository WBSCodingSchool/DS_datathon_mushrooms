import logging
import sys

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import get_global_store


class CloudLogFormatter(logging.Formatter):
    # ANSI Terminal Palette Codes
    RESET = "\033[0m"
    ORANGE = "\033[33m"
    GREEN = "\033[32m"
    MAX_USER_LENGTH = 10

    def format(self, record):
        level_map = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARN",
            "ERROR": "ERROR",
            "CRITICAL": "FATAL",
        }
        raw_user = str(getattr(record, "user", "SYSTEM"))
        user_formatted = (
            raw_user[:self.MAX_USER_LENGTH]
            if len(raw_user) > self.MAX_USER_LENGTH
            else raw_user.ljust(self.MAX_USER_LENGTH)
        )


        asctime = self.formatTime(record, self.datefmt)
        levelname = level_map.get(record.levelname, f"{record.levelname:<5}")
        msg = record.getMessage()

        if "waitlisted" in msg.lower():
            color_prefix = self.ORANGE
        elif "completed" in msg.lower() or "success" in msg.lower():
            color_prefix = self.GREEN
        else:
            color_prefix = ""

        # Assemble the final log stream grid string
        if color_prefix:
            return f"{asctime} {levelname} - {user_formatted} {color_prefix}{msg}{self.RESET}"
        return f"{asctime} {levelname} - {user_formatted} {msg}"

log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(CloudLogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [log_handler]


def get_ready_test(results_path: str, uploaded_file) -> pd.DataFrame:
    """This function prepares the test file to be evaluated and
    check if it has the correct format.
    """
    results = pd.read_csv(results_path)
    results.columns = ["id", "real"]

    test = pd.read_csv(uploaded_file)
    if test.columns.to_list() != ["Id", "poisonous"]:
        st.error('Column names must match "Id" and "poisonous" - case sensitive!')
        return 0
    if test.shape != (1625, 2):
        st.error("Your file should contain 1625 rows and 2 columns")
        return 0
    if (
        (test.poisonous.unique().tolist() != [0, 1])
        & (test.poisonous.unique().tolist() != [1, 0])
        & (test.poisonous.unique().tolist() != [1])
        & (test.poisonous.unique().tolist() != [0])
    ):
        st.error("Predictions should only have values of 0 and 1")
        return 0
    if (test.Id == results.id).sum() != 1625:
        st.error(
            "Your Id column might be wrong or mixed up. "
            "You should have same Id's as the test file. "
            "Order of Id's should also be the same.",
        )
        return 0
    test.columns = ["id", "preds"]

    return test.astype("int32")


def get_metrics(results_path: str, test: pd.DataFrame) -> pd.DataFrame:
    """Calculates metrics and prepares a single-row DataFrame for GSheets submission."""
    results = pd.read_csv(results_path)
    results.columns = ["id", "real"]

    row_evaluation = (
        results.astype("int32")
        .merge(test, how="left", on="id")
        .assign(
            tp=lambda df_: np.where(
                (df_["real"] == 1) & (df_["preds"] == 1),
                True,
                False,
            ),
            correct=lambda df_: df_["real"] == df_["preds"],
            fn=lambda df_: np.where(
                (df_["real"] == 1) & (df_["preds"] == 0),
                True,
                False,
            ),
            opportunity_cost=lambda df_: np.where(
                (df_["real"] == 0) & (df_["preds"] == 1),
                True,
                False,
            ),
        )
        .agg(
            {
                "tp": "sum",
                "correct": "sum",
                "fn": "sum",
                "opportunity_cost": "sum",
            },
        )
    )

    score = round(
        row_evaluation["tp"]
        / (row_evaluation["tp"] + row_evaluation["fn"])
        * 0.95
        + row_evaluation["correct"] / results.shape[0] * 0.05,
        4,
    )
    recall = round(
        row_evaluation["tp"]
        / (row_evaluation["tp"] + row_evaluation["fn"]),
        4,
    )
    accuracy = round(row_evaluation["correct"] / results.shape[0], 4)
    hospitalized = int(row_evaluation["fn"])
    edible = int(row_evaluation["opportunity_cost"])

    logger.info(
        f"Evaluation successfull. Accuracy: {score} ({hospitalized} hosp., {edible} uneaten)",
        extra={"user": st.session_state.user_name, "comp": "UPLOADER"},
    )

    return pd.DataFrame(
        [
            {
                "Participant": st.session_state.user_name,
                "Scoring metric": score,
                "Recall": recall,
                "Accuracy": accuracy,
                "Hospitalized": hospitalized,
                "Edible but uneaten": edible,
                "submission_time": pd.Timestamp.now().isoformat(),
                "batch": st.session_state.batch,
            },
        ],
    )


def update_submissions(participant_results: pd.DataFrame) -> None:
    """Writes results to GSheets and updates memory store to avoid re-reading."""
    store = get_global_store()
    batch = st.session_state.batch

    current_submissions = store.setdefault("submissions", {}).get(batch, pd.DataFrame())
    updated_df = pd.concat(
        [current_submissions, participant_results],
        ignore_index=True,
    )
    store["submissions"][batch] = updated_df

    try:
        store["gsheet_conn"].update(worksheet=batch, data=updated_df)

        if batch != "anonymous" and store.get("alltime_submissions") is not None:
            store["alltime_submissions"] = pd.concat(
                [store["alltime_submissions"], participant_results],
                ignore_index=True,
            )
    except Exception as e:
        st.error(f"Sync failed: {e}")
