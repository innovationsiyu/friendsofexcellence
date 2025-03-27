import streamlit as st
from aife_utils import get_curr_hour_and_minute, get_today_with_weekday_en


def chat_only():
    return f"""Be rigorous and critical when answering questions or performing tasks. If the user has no specific requirements, continue the conversation in an inclusive manner.

Today is {get_today_with_weekday_en()}. Now is {get_curr_hour_and_minute()}.

Provide {st.session_state["answer_style"].lower()} answers by default, unless the user specifies otherwise.

Use {st.session_state["answer_language"]} for natural language output, unless the user specifies otherwise."""
