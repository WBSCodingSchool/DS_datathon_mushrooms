import pandas as pd
import streamlit as st

from src.gsheet import ensure_batch_sheet_exists
from src.pd_functions import get_metrics, get_ready_test, update_submissions
from src.utils import RESULTS_PATH, get_global_store


def plot_submissions() -> None:
    """Plots the participant's progress using data from the global store."""
    store = get_global_store()
    batch = st.session_state.batch
    user = st.session_state.user_name

    if batch not in store["submissions"]:
        return

    participant_submissions = (
        store["submissions"][batch]
        .query("Participant == @user")
        .filter(["submission_time", "Scoring metric"])
        .copy()
    )

    if len(participant_submissions) > 1:
        st.divider()
        st.subheader("📊 Your progress over time")
        participant_submissions["submission_time"] = pd.to_datetime(
            participant_submissions["submission_time"],
        )
        participant_submissions = participant_submissions.sort_values(
            "submission_time",
        ).set_index("submission_time")
        st.line_chart(participant_submissions)
    elif len(participant_submissions) == 1:
        st.success("Congratulations on your first submission!")
    else:
        st.warning("Submit your first submission to join the competition!")


def display_participant_results(participant_results) -> None:
    st.title("Participant results")
    st.dataframe(participant_results)


def get_participant_info() -> bool:
    """Handles login and validates the secret batch code."""
    store = get_global_store()

    if st.session_state.user_name and st.session_state.batch:
        if st.session_state.batch not in store["submissions"]:
            try:
                # 1. Ensure the tab exists in the physical Sheet
                ensure_batch_sheet_exists(st.session_state.batch)

                # 2. Load the data into the Global Store
                store["submissions"][st.session_state.batch] = store[
                    "gsheet_conn"
                ].read(worksheet=st.session_state.batch, ttl=0)
            except Exception as e:
                st.error(f"Error loading batch data: {e}")
                st.stop()

        st.info(
            f"Logged in as: **{st.session_state.user_name}** | Batch: **{st.session_state.batch}**",
        )
        return True

    st.subheader("🔑 Batch Login")

    time_since_update = 999999
    if store["batches_last_updated"] is not None:
        time_since_update = (
            pd.Timestamp.now() - store["batches_last_updated"]
        ).total_seconds()
    if store["batches"] is None or time_since_update > 600:
        try:
            store["batches"] = store["gsheet_conn"].read(worksheet="Batches", ttl=0)
            store["batches_last_updated"] = pd.Timestamp.now()
        except Exception:
            st.error("Could not connect to the Batch Database.")
            st.stop()

    st.write(
        "If you haven't done so yet, please download the test data, train data, and an example upload file below.",
    )

    cols = st.columns(3)

    with cols[0], open("data/test.csv", "rb") as f:
        st.download_button(
            label="Download test data",
            data=f,
            file_name="test.csv",
            mime="text/csv",
        )

    with cols[1], open("data/train.csv", "rb") as f:
        st.download_button(
            label="Download train data",
            data=f,
            file_name="train.csv",
            mime="text/csv",
        )

    with cols[2], open("data/sample_submission.csv", "rb") as f:
        st.download_button(
            label="Download example upload",
            data=f,
            file_name="example_upload.csv",
            mime="text/csv",
        )

    st.divider()

    st.warning(
        "Please enter **your name** (real or alias) and **the code** provided by your instructor.",
    )

    user_name = st.text_input("Username (Name or Alias):")
    code_input = st.text_input("Secret Batch Code:", type="password")

    if user_name and code_input:
        batches_df = store["batches"]
        if code_input in batches_df["Code"].to_numpy():
            row = batches_df[batches_df["Code"] == code_input].iloc[0]

            st.session_state.user_name = user_name
            st.session_state.batch = row["Batch"]

            st.session_state.alltime = (
                False if pd.isna(row["Show All-time?"]) else row["Show All-time?"]
            )

            st.rerun()
        else:
            st.error("Invalid Code. Please check with your instructor.")
    return False


def display_admin() -> None:
    """Admin tools for instructors to clear global cache."""
    st.divider()
    st.subheader("🛠️ Instructor Settings")
    if st.button("Clear Global Cache"):
        get_global_store().clear()
        st.success("Cache cleared! Refreshing...")
        st.rerun()


@st.fragment(run_every=10)
def show_leaderboard() -> None:
    """Displays the leaderboard using the in-memory store."""
    if st.session_state.batch == "anonymous":
        st.info(
            "You are currently in an anonymous session. Your results are being recorded for instructors, "
            "but you won't see or appear on any public leaderboards.",
        )
        return

    store = get_global_store()
    batch = st.session_state.batch

    df = store.get("submissions", {}).get(batch, pd.DataFrame())
    st.divider()
    st.header(f"🏆 {batch} Leaderboard")
    if not df.empty:
        leaderboard_df = (
            df.assign(
                Attempts=lambda df_: df_.groupby("Participant")[
                    "Participant"
                ].transform("count"),
            )
            .sort_values(
                ["Scoring metric", "Recall", "Accuracy"],
                ascending=[False, False, False],
            )
            .drop_duplicates(["Participant"], keep="first")
            .assign(
                position=lambda df_: df_["Scoring metric"]
                .rank(method="min", ascending=False)
                .astype(int),
            )
            .assign(
                position=lambda df_: df_["position"]
                .where(df_["position"].ne(df_["position"].shift()), "")
                .astype(str),
            )
            .set_index("position")
            .filter(["Participant", "Scoring metric", "Recall", "Accuracy",
                     "Hospitalized", "Edible but uneaten", "Attempts"])
        )
        st.dataframe(leaderboard_df, use_container_width=True)
    else:
        st.info("No submissions yet for this batch.")

    st.session_state["manual_refresh"] = False
    if st.button("Refresh leaderboard(s)"):
        st.session_state["manual_refresh"] = True
        st.rerun()

    if st.session_state.alltime and store.get("alltime_submissions") is not None:
        at_df = store["alltime_submissions"]
        if not at_df.empty:
            st.divider()
            st.header("👑 All-time Global Leaderboard", anchor=False)
            st.dataframe(
                at_df.assign(
                    Attempts=lambda df_: df_.groupby("Participant")[
                        "Participant"
                    ].transform("count"),
                )
                .sort_values("Scoring metric", ascending=False)
                .drop_duplicates(["Participant"], keep="first")
                .assign(
                    position=lambda d: d["Scoring metric"]
                    .rank(method="min", ascending=False)
                    .astype(int),
                )
                .assign(
                    position=lambda d: d["position"]
                    .where(d["position"].ne(d["position"].shift()), "")
                    .astype(str),
                )
                .set_index("position")
                .filter(
                    [
                        "Participant",
                        "Scoring metric",
                        "Recall",
                        "Accuracy",
                        "Hospitalized",
                        "Edible but uneaten",
                        "batch",
                        "Attempts",
                    ],
                ),
                use_container_width=True,
            )


def display_upload_and_evaluate() -> None:
    uploaded_file = st.file_uploader("Choose your submission CSV file", type="csv")

    if uploaded_file and not st.session_state.get("manual_refresh", False):
        try:
            # Prepare and validate the data
            test = get_ready_test(RESULTS_PATH, uploaded_file)

            if isinstance(test, pd.DataFrame):
                # Calculate scores
                participant_results = get_metrics(RESULTS_PATH, test)

                poisoned = participant_results["Hospitalized"].to_numpy()[0]
                edible = participant_results["Edible but uneaten"].to_numpy()[0]
                info_msg = "Thank you for your submission."
                if poisoned > 0:
                    info_msg = (
                        info_msg
                        + f" \n \n You poisened in total {poisoned} people:\n \n "
                        + "🤢" * poisoned
                    )
                if edible > 0:
                    info_msg = (
                        info_msg
                        + f" \n \n In total {edible} edible mushrooms were not eaten:\n \n "
                        + "🍄‍🟫" * edible
                    )

                st.warning(info_msg)

                st.dataframe(participant_results)

                update_submissions(participant_results)

        except Exception as e:
            st.error(f"Error processing file: {e!s}")
