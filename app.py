import streamlit as st

from src.display import (
    display_admin,
    display_upload_and_evaluate,
    get_participant_info,
    plot_submissions,
    show_leaderboard,
)
from src.utils import state_inits


def main() -> None:
    st.set_page_config(
        page_title="Mushroom Classification Challenge",
        page_icon="🍄",
        layout="wide",
    )
    st.title("🍄 Mushroom Classification Challenge", anchor=False)

    if st.session_state.get("batch") == "Instructor":
        display_admin()

    state_inits()

    if get_participant_info():
        st.subheader("📤 Submit Your Predictions", anchor=False)

        display_upload_and_evaluate()

        plot_submissions()
        show_leaderboard()


if __name__ == "__main__":
    main()
