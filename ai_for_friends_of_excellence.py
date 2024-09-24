from aife_dict import aife_dict
import streamlit as st
import os
import datetime

st.session_state["ai"] = st.query_params.get("ai", st.session_state.get("ai", "yusi_mini"))
st.session_state["latest_result"] = st.session_state.get("latest_result", "")
st.session_state["chat_history"] = st.session_state.get("chat_history", "")

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; font-size: 24px;'>优秀的朋友用的 AI for Friends of Excellence</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='font-size: 14px; max-width: 800px; margin: auto; text-align: justify;'>
<p>Here, you can access the leading large language models without any special network setup, tailored specifically for Siyu's friends of excellence. The interface is designed according to my personal preferences, featuring support for interacting with multiple models in a single thread, freely editing and downloading all results, and easily referring to chat history while composing the current message.</p>

<p style='margin-bottom: 24px;'>Our services include tailoring automated workflows and AI agents for businesses and individuals. We would like you to join "思宇的优秀的朋友的群". Add me on WeChat at "innovationsiyu".</p>
</div>
""", unsafe_allow_html=True)


def select():
    st.session_state["ai"] = st.session_state["ai"]
    st.query_params.update({"ai": st.session_state["ai"]})


def text_chat(user_message):
    user_message = user_message.strip()
    user_message_with_history = f"{st.session_state["chat_history"]}\n\nUser:\n{user_message}"

    if len(user_message_with_history) > 100000:
        return "The messages are too long. You can shorten the chat history."

    ai = st.session_state["ai"]
    function = aife_dict[ai]["function"]
    system_message = aife_dict[ai]["system_message"]

    with st.spinner("Let me think... 🧠"):
        response = function(system_message, user_message_with_history)

    st.session_state["latest_result"] = f"User:\n{user_message}\n\nAssistant ({ai}):\n{response}\n\n"
    st.session_state["chat_history"] += f"User:\n{user_message}\n\nAssistant ({ai}):\n{response}\n\n"


def image_chat(user_message, image_paths):
    user_message = user_message.strip()
    user_message_with_history = f"{st.session_state["chat_history"]}\n\nUser:\n{user_message}"

    if len(user_message_with_history) > 100000:
        return "The messages are too long. You can shorten the chat history."

    ai = st.session_state["ai"]
    function = aife_dict[ai]["function"]

    responses = []
    for image_path in image_paths:
        with st.spinner(f"Let me see... 👀 {image_path}"):
            response = function(user_message_with_history, image_path)
            if response:
                responses.append(response)

        if os.path.isfile(image_path):
            os.remove(image_path)

    st.session_state["latest_result"] = f"User:\n{user_message}\n\nAssistant ({ai}):\n{'\n'.join(responses)}\n\n"
    st.session_state["chat_history"] += f"User:\n{user_message}\n\nAssistant ({ai}):\n{'\n'.join(responses)}\n\n"


with st.sidebar:
    ai_options = list(aife_dict.keys())
    st.radio("AI options", ai_options, key="ai", on_change=select, label_visibility="collapsed")

user_message_column, latest_result_column = st.columns([1, 1])

with user_message_column:
    with st.form(key="user_message_form", clear_on_submit=True):
        st.markdown("<p style='font-size: 14px; margin-top: 4px;'>Input a message</p>", unsafe_allow_html=True)
        user_message = st.text_area("Input a message", height=500, key="user_message", label_visibility="collapsed")
        send_message = st.form_submit_button(label="Send message", use_container_width=True)
        images = st.file_uploader("Upload images to use the vision model", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if send_message and user_message.strip():
        ai = st.session_state["ai"]
        if aife_dict[ai]["category"] == "vision":
            if images:
                image_paths = []
                for image in images:
                    image_path = image.name
                    with open(image_path, "wb") as f:
                        f.write(image.getbuffer())
                    image_paths.append(image_path)
                image_chat(user_message, image_paths)
            else:
                st.error("Please upload an image to use the vision model")
        else:
            text_chat(user_message)

timestamp_second = datetime.datetime.now().strftime("%Y%m%d %H%M%S")

with latest_result_column:
    latest_result_tab, chat_history_tab = st.tabs(["Latest result (editable)", "Chat history (editable)"])

    with latest_result_tab:
        st.text_area("Latest result (editable)", height=500, key="latest_result", label_visibility="collapsed")
        st.download_button("Download latest result", st.session_state["latest_result"], f"Latest result {timestamp_second}.txt", "text/plain", use_container_width=True)

    with chat_history_tab:
        st.text_area("Chat history (editable)", height=500, key="chat_history", label_visibility="collapsed")
        st.download_button("Download chat history", st.session_state["chat_history"], f"Chat history {timestamp_second}.txt", "text/plain", use_container_width=True)

st.components.v1.html(
    """
    <script>
    let scrollTimeout;
    let shouldScrollNext = false;

    function scrollChatHistoryToBottom() {
        const chatHistoryTextArea = parent.document.querySelector('textarea[aria-label="Chat history (editable)"]');
        if (chatHistoryTextArea) {
            chatHistoryTextArea.scrollTop = chatHistoryTextArea.scrollHeight;
        }
    }

    function checkAndScroll() {
        if (shouldScrollNext) {
            scrollChatHistoryToBottom();
            shouldScrollNext = false;
        }
    }

    function delayedCheck() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(checkAndScroll, 100);
    }

    // 监听模型响应区域的变化
    const latestResultObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' || mutation.type === 'characterData') {
                const latestResultObserver = parent.document.querySelector('textarea[aria-label="Latest result (editable)"]');
                if (latestResultObserver && mutation.target.contains(latestResultObserver)) {
                    shouldScrollNext = true;
                }
            }
        });
    });

    // 监听聊天历史区域的变化
    const chatHistoryObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' || mutation.type === 'characterData') {
                const chatHistoryTextArea = parent.document.querySelector('textarea[aria-label="Chat history (editable)"]');
                if (chatHistoryTextArea && mutation.target.contains(chatHistoryTextArea)) {
                    delayedCheck();
                }
            }
        });
    });

    // 配置观察器
    const config = { childList: true, subtree: true, characterData: true };

    // 开始观察
    latestResultObserver.observe(parent.document.body, config);
    chatHistoryObserver.observe(parent.document.body, config);
    </script>
    """,
    height=0,
)
