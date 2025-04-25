import streamlit as st
import os
import re
import hashlib
import tempfile
from pathvalidate import sanitize_filename
import copy
import base64
import traceback
import pyperclip
from aife_tools import get_prompt, get_tools, route_llm_request, speech_to_text, jobs, llms, parse_pdfs
from aife_utils import resize_images, get_text, ensure_utf_8_csvs, now_and_choices


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.components.v1.html(
    """<script>
    const updateWindowHeight = () => {
        const windowHeight = window.parent.innerHeight;
        document.cookie = `window_height=${windowHeight};path=/`;
    };
    updateWindowHeight();
    window.parent.addEventListener('resize', updateWindowHeight);
    </script>""",
    height=0
)

window_height = int(st.context.cookies.get("window_height", 830))
print(window_height)

st.session_state["timezone"] = st.session_state.get("timezone", st.context.timezone or "Asia/Shanghai")
st.session_state["job"] = st.query_params.get("job", st.session_state.get("job", "Text Chat"))
st.session_state["llm"] = st.session_state.get("llm", jobs[st.session_state["job"]]["llms"][0])
st.session_state["temperature"] = float(st.query_params.get("temperature", st.session_state.get("temperature", 0.5)))
st.session_state["top_p"] = float(st.query_params.get("top_p", st.session_state.get("top_p", 0.95)))
st.session_state["answer_language"] = st.session_state.get("answer_language", "简体中文")
st.session_state["pdf_type"] = st.session_state.get("pdf_type", "Plain Text")
st.session_state["messages"] = st.session_state.get("messages", [])
st.session_state["last_audio_hash"] = st.session_state.get("last_audio_hash")


def get_file_paths(files):
    file_paths = []
    for file in files:
        try:
            file_name = file.name
            file_path = f"uploaded-files/{sanitize_filename(os.path.splitext(file_name)[0])} {now_and_choices()}{os.path.splitext(file_name)[1]}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        except Exception:
            continue
    return file_paths


def select_job():
    st.query_params.update({"job": st.session_state["job"]})
    st.session_state["llm"] = jobs[st.session_state["job"]]["llms"][0]


def set_temperature():
    st.query_params.update({"temperature": st.session_state["temperature"]})


def set_top_p():
    st.query_params.update({"top_p": st.session_state["top_p"]})


def transcribe_user_message(user_message_audio):
    buffer = user_message_audio.getbuffer()
    current_audio_hash = hashlib.md5(buffer[:min(len(buffer), 882000)]).hexdigest()
    if current_audio_hash != st.session_state["last_audio_hash"]:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(buffer)
        try:
            if user_message := speech_to_text(f.name):
                return user_message
        except Exception:
            print(f"An error occurred:\n{traceback.format_exc()}")
        finally:
            st.session_state["last_audio_hash"] = current_audio_hash
            os.remove(f.name)
    return None


def append_user_message(user_message, file_paths):
    try:
        if file_paths:
            if pdf_paths := [file_path for file_path in file_paths if file_path.endswith(".pdf")]:
                contents, txt_path = parse_pdfs(pdf_paths, st.session_state["pdf_type"], True)
                st.session_state["messages"].append({"role": "user", "content": f"<doc_content>{'\n\n'.join(contents)}</doc_content>\n\nThe path to the TXT file containing the doc_content is '{txt_path}'"})
            if txt_paths := [file_path for file_path in file_paths if file_path.endswith(".txt")]:
                texts, txt_path = get_text(txt_paths, True)
                st.session_state["messages"].append({"role": "user", "content": f"<doc_content>{'\n\n'.join(texts)}</doc_content>\n\nThe path to the TXT file containing the doc_content is '{txt_path}'"})
            if table_paths := [file_path for file_path in file_paths if file_path.endswith((".csv", ".xlsx", ".xls"))]:
                table_paths = ensure_utf_8_csvs(table_paths)
                st.session_state["messages"].append({"role": "user", "content": f"The paths to the CSV files are: '{table_paths}'"})
            if (image_paths := [file_path for file_path in file_paths if file_path.endswith((".jpg", ".jpeg", ".png"))]) and llms[st.session_state["llm"]]["category"] == "multimodal":
                resize_images(image_paths, 1280, False)
                st.session_state["messages"].append({"role": "user", "content": user_message, "image_paths": image_paths})
            else:
                st.session_state["messages"].append({"role": "user", "content": user_message})
        else:
            st.session_state["messages"].append({"role": "user", "content": user_message})
    except Exception:
        print(f"An error occurred:\n{traceback.format_exc()}")


def sync_messages(index=None):
    if index is not None:
        messages = st.session_state["messages"][:index + 1]
    else:
        messages = st.session_state["messages"]
    return "\n\n\n\n".join([f"{'User' if message['role'] == 'user' else f'AI ({st.session_state["llm"]})'}:\n{message['content']}{('\n' + '\n'.join(message['image_paths'])) if 'image_paths' in message else ''}" for message in messages])


def update_messages(chat_history):
    pattern = r"(User:\n|AI \([^)]+\):\n)"
    segments = ["User:\n"] + [segment for segment in re.split(pattern, chat_history) if segment.strip()]
    messages = []
    for i in range(len(segments) - 1):
        if re.match(pattern, segments[i]) and not re.match(pattern, segments[i + 1]):
            role = "user" if segments[i] == "User:\n" else "assistant"
            content = segments[i + 1].strip()
            message = {"role": role, "content": content}
            if (content.endswith((".jpg", ".jpeg", ".png")) and ("uploaded-files/" in content or "temp-images/" in content)):
                image_paths = []
                while (content.rfind("\n") > 0 and (last_line := content[content.rfind("\n") + 1:]).startswith(("uploaded-files/", "temp-images/")) and last_line.endswith((".jpg", ".jpeg", ".png"))):
                    image_paths.insert(0, last_line)
                    content = content[:content.rfind("\n")].rstrip()
                if image_paths:
                    message.update({"content": content, "image_paths": image_paths})
            messages.append(message)
    st.session_state["messages"] = messages


def interleave_messages(messages):
    for i in reversed(range(1, len(messages))):
        last_message, current_message = messages[i - 1], messages[i]
        if last_message["role"] == current_message["role"]:
            last_message["content"] += "\n\n" + current_message["content"]
            if "image_paths" in current_message:
                if "image_paths" in last_message:
                    last_message["image_paths"].extend(current_message["image_paths"])
                else:
                    last_message["image_paths"] = current_message["image_paths"].copy()
            messages.pop(i)
    return messages


def get_assistant_message(continue_message=None, retry=0):
    try:
        messages = interleave_messages(copy.deepcopy(st.session_state["messages"]))
        for i, message in enumerate(messages):
            if "image_paths" in message:
                if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                    messages[i] = {"role": message["role"], "content": [{"type": "text", "text": message["content"]}, *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}} for image_path in image_paths]]}
        if continue_message:
            messages.append({"role": "user", "content": continue_message})
        selected_job = jobs[st.session_state["job"]]
        messages.insert(0, {"role": "system", "content": get_prompt(selected_job["system_message"])})
        selected_llm = llms[st.session_state["llm"] if st.session_state["llm"] else jobs[st.session_state["job"]]["llms"][0]]
        llm_result = route_llm_request(selected_llm["configs"], messages, st.session_state["temperature"], st.session_state["top_p"], get_tools(selected_job["tools"]))
        if isinstance(llm_result, str):
            st.session_state["messages"].append({"role": "assistant", "content": llm_result})
            st.rerun()
        elif isinstance(llm_result, dict) and llm_result["type"] == "tool_results":
            st.session_state["messages"].append({"role": "assistant", "content": llm_result["tool_results"]})
            get_assistant_message(get_prompt("reply_with_tool_results" if retry < 5 else "reply_with_tool_results2"), retry + 1)
        else:
            error_message = f"I can only handle a context length of {selected_llm['context_length']} tokens. The total length of all messages possibly exceeded my limit. Please start a new conversation, or cut parts of the chat history, or continue with other LLMs such as Gemini, which can process up to 1 million tokens."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
    except Exception:
        print(f"An error occurred:\n{traceback.format_exc()}")


def style_message_content(content):
    if matched := re.search(r"^<think>([\s\S]*)</think>\s*([\s\S]*)", content):
        if reasoning := re.compile(r"\n+").sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{reasoning}</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<doc_content>([\s\S]*)</doc_content>\s*([\s\S]*)", content):
        if doc_content := re.compile(r"\n+").sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{doc_content[:300]}...</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<tool_results>([\s\S]*)</tool_results>\s*([\s\S]*)", content):
        return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{matched.group(1)[:300]}...</div>{matched.group(2)}"
    return content


def display_messages(index):
    messages = st.session_state["messages"][index:]
    for i, message in enumerate(messages, start=index):
        with st.container(border=True):
            if message["role"] == "user":
                with st.chat_message("user", avatar=":material/face:"):
                    st.write(f"User:\n{style_message_content(message['content'])}", unsafe_allow_html=True)
                    if "image_paths" in message:
                        if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                            st.image(image_paths, width=240)
            elif message["role"] == "assistant":
                with st.chat_message("assistant", avatar=":material/smart_toy:"):
                    st.write(f"AI ({st.session_state['llm']}):\n{style_message_content(message['content'])}", unsafe_allow_html=True)
                    if "image_paths" in message:
                        if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                            st.image(image_paths, use_container_width=True)
            left_space, copy_message_button, edit_message_button, delete_message_button, right_space = st.columns(5)
            with copy_message_button:
                if st.button(":material/content_copy:", key=f"copy_message_{i}", help="Copy this message", type="tertiary", use_container_width=True):
                        pyperclip.copy(message["content"])
                        st.toast("Copied")
            with edit_message_button:
                if st.button(":material/edit_document:", key=f"edit_message_{i}", help="Edit messages up to this", type="tertiary", use_container_width=True):
                    edit_messages(i)
            with delete_message_button:
                if st.button(":material/remove:", key=f"delete_message_{i}", help="Delete this message", type="tertiary", use_container_width=True):
                    st.session_state["messages"].pop(i)
                    st.rerun()


@st.dialog("Edit chat history", width="large")
def edit_messages(index):
    chat_history = st.text_area("Edit chat history", sync_messages(index), height=window_height - 236, label_visibility="collapsed")
    if st.button("Fork a new chat", type="tertiary", icon=":material/arrow_split:", use_container_width=True):
        update_messages(chat_history)
        st.rerun()

st.components.v1.html(
    """<script>
    new MutationObserver(function() {
        const textarea = window.parent.document.querySelector('div[data-testid="stDialog"] textarea');
        if (textarea) {
            textarea.scrollTop = textarea.scrollHeight;
        }
    }).observe(window.parent.document.body, { childList: true, subtree: true });
    </script>""",
    height=0
)


def chat(user_message, file_paths=[]):
    append_user_message(user_message, file_paths)
    display_messages(0)
    displayed_quantity = len(st.session_state["messages"])
    get_assistant_message()
    display_messages(displayed_quantity)


chat_input = st.chat_input("Input a message", accept_file="multiple", file_type=["pdf", "txt", "csv", "xlsx", "xls", "jpg", "jpeg", "png"])
user_message = chat_input.text.strip() if chat_input else None
file_paths = get_file_paths(chat_input.files if chat_input else [])

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 16px;'><a href='https://friendsofexcellence.ai' style='text-decoration: none; color: inherit;'>优秀的朋友用的 AI for Friends of Excellence</a></h1>", unsafe_allow_html=True)

    with st.container(height=(window_height - 400) // 2, border=False):
        job_options = list(jobs.keys())
        captions = [jobs[job_option]["intro"] for job_option in job_options]
        st.radio("Job Options", job_options, key="job", on_change=select_job, captions=captions, label_visibility="collapsed")

    with st.popover("Params", use_container_width=True):
        st.slider("Temperature", 0.0, 1.0, key="temperature", on_change=set_temperature)
        st.slider("Top P", 0.01, 1.0, key="top_p", on_change=set_top_p)
        pdf_types = ["Plain Text", "With Visuals"]
        st.pills("PDF Type", pdf_types, key="pdf_type")

    with st.container(height=(window_height - 488) // 2, border=False):
        llm_options = jobs[st.session_state["job"]]["llms"]
        st.pills("LLM Options", llm_options, key="llm", label_visibility="collapsed")

    if user_message_audio := st.audio_input("Click the mic and speak", label_visibility="collapsed"):
        user_message = transcribe_user_message(user_message_audio)

    st.download_button("Export chat history", sync_messages(), f"Messages {now_and_choices()}.txt", "text/plain", type="tertiary", use_container_width=True)

if user_message:
    chat(user_message, file_paths)
else:
    display_messages(0)
