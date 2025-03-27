import streamlit as st
import os
import re
import hashlib
import tempfile
from pathvalidate import sanitize_filename
import copy
import base64
import traceback
from aife_tools import get_prompt, get_tools, route_llm_request, speech_to_text, jobs, llms, parse_pdfs
from aife_utils import RE_COMPRESS_NEWLINES, resize_images, ensure_utf_8_csvs, now_and_choices


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.components.v1.html(
    """
    <script>
    const updateWindowHeight = () => {
        const windowHeight = window.parent.innerHeight;
        document.cookie = `window_height=${windowHeight};path=/`;
    };

    updateWindowHeight();

    window.parent.addEventListener('resize', updateWindowHeight);
    </script>
    """,
    height=0
)

window_height = int(st.context.cookies.get("window_height", 830))

st.session_state["timezone"] = st.session_state.get("timezone", st.context.timezone or "Asia/Shanghai")
st.session_state["job"] = st.query_params.get("job", st.session_state.get("job", "Text Chat"))
st.session_state["llm"] = st.session_state.get("llm", jobs[st.session_state["job"]]["llms"][0])
st.session_state["temperature"] = float(st.query_params.get("temperature", st.session_state.get("temperature", 0.5)))
st.session_state["top_p"] = float(st.query_params.get("top_p", st.session_state.get("top_p", 0.95)))
st.session_state["answer_style"] = st.session_state.get("answer_style", "Concise Conclusive")
st.session_state["answer_language"] = st.session_state.get("answer_language", "简体中文")
st.session_state["pdf_type"] = st.session_state.get("pdf_type", "Plain Text")
st.session_state["messages"] = st.session_state.get("messages", [])
st.session_state["revised_messages"] = st.session_state.get("revised_messages", "")
st.session_state["last_audio_hash"] = st.session_state.get("last_audio_hash")


def get_file_name_and_path_tuples(files):
    file_name_and_path_tuples = []
    for file in files:
        try:
            file_name = file.name
            file_path = f"uploaded-files/{sanitize_filename(os.path.splitext(file_name)[0])} {now_and_choices()}{os.path.splitext(file_name)[1]}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_name_and_path_tuples.append((file_name, file_path))
        except Exception:
            continue
    return file_name_and_path_tuples


def get_pdfs(file_name_and_path_tuples):
    pdf_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith((".pdf"))]
    pdf_names = [file_name for file_name, file_path in pdf_name_and_path_tuples]
    pdf_paths = [file_path for file_name, file_path in pdf_name_and_path_tuples]
    return pdf_names, pdf_paths


def get_tables(file_name_and_path_tuples):
    table_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith((".csv", ".xlsx", ".xls"))]
    table_names = [file_name for file_name, file_path in table_name_and_path_tuples]
    if table_paths := [file_path for file_name, file_path in table_name_and_path_tuples]:
        table_paths = ensure_utf_8_csvs(table_paths)
    return table_names, table_paths


def get_images(file_name_and_path_tuples):
    image_name_and_path_tuples = [(file_name, file_path) for file_name, file_path in file_name_and_path_tuples if file_path.endswith((".jpg", ".jpeg", ".png"))]
    image_names = [file_name for file_name, file_path in image_name_and_path_tuples]
    if image_paths := [file_path for file_name, file_path in image_name_and_path_tuples]:
        resize_images(image_paths, 1280, False)
    return image_names, image_paths


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
            st.warning(f"An error occurred:\n{traceback.format_exc()}")
        finally:
            st.session_state["last_audio_hash"] = current_audio_hash
            os.remove(f.name)
    return None


def append_user_message(user_message, file_name_and_path_tuples):
    try:
        if file_name_and_path_tuples:
            pdf_names, pdf_paths = get_pdfs(file_name_and_path_tuples)
            if pdf_names and pdf_paths:
                pdf_contents, txt_path = parse_pdfs(pdf_paths, st.session_state["pdf_type"])
                st.session_state["messages"].append({"role": "user", "content": f"<doc_content>{'\n\n'.join(pdf_contents[key] for key in sorted(pdf_contents))}</doc_content>\n\nThe path of the TXT file containing the doc_content: {txt_path}", "file_names": pdf_names})
            table_names, table_paths = get_tables(file_name_and_path_tuples)
            if table_names and table_paths:
                st.session_state["messages"].append({"role": "user", "content": f"The paths of the CSV files: {table_paths}", "file_names": table_names})
            image_names, image_paths = get_images(file_name_and_path_tuples)
            if image_names and image_paths and jobs[st.session_state["job"]]["category"] == "multimodal":
                st.session_state["messages"].append({"role": "user", "content": user_message, "file_names": image_names, "image_paths": image_paths})
            else:
                st.session_state["messages"].append({"role": "user", "content": user_message})
        else:
            st.session_state["messages"].append({"role": "user", "content": user_message})
    except Exception:
        st.warning(f"An error occurred:\n{traceback.format_exc()}")


def update_messages():
    pattern = r"(User:\n|AI \([^)]+\):\n)"
    segments = ["User:\n"] + [segment for segment in re.split(pattern, st.session_state["revised_messages"]) if segment.strip()]
    return [{"role": "user" if segments[i] == "User:\n" else "assistant", "content": segments[i + 1].strip()} for i in range(len(segments) - 1) if re.match(pattern, segments[i]) and not re.match(pattern, segments[i + 1])]


def sync_messages():
    return "\n\n\n\n".join([f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}" for message in st.session_state["messages"]])


def interleave_messages(messages):
    for i in reversed(range(1, len(messages))):
        last_message, current_message = messages[i - 1], messages[i]
        if last_message["role"] == current_message["role"] and isinstance(current_message["content"], str):
            last_message["content"] += "\n\n" + current_message.pop("content")
            messages.pop(i)
    return messages


def chat(continue_message=None, retry=0):
    try:
        messages = copy.deepcopy(st.session_state["messages"])
        for i, message in enumerate(messages):
            if "image_paths" in message:
                if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                    messages[i]["content"] = [{"type": "text", "text": message["content"]}, *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}} for image_path in image_paths]]
        if continue_message:
            messages.append({"role": "user", "content": continue_message})
        selected_job = jobs[st.session_state["job"]]
        messages = [{"role": "system", "content": get_prompt(selected_job["system_message"])}] + interleave_messages(messages)
        selected_llm = llms[st.session_state["llm"] if st.session_state["llm"] else jobs[st.session_state["job"]]["llms"][0]]
        llm_result = route_llm_request(selected_llm["configs"], messages, st.session_state["temperature"], st.session_state["top_p"], get_tools(selected_job["tools"]))
        if isinstance(llm_result, str):
            st.session_state["messages"].append({"role": "assistant", "content": llm_result})
            st.session_state["revised_messages"] = sync_messages()
            st.rerun()
        elif isinstance(llm_result, dict) and llm_result["type"] == "tool_results":
            st.session_state["messages"].append({"role": "assistant", "content": llm_result["tool_results"]})
            chat(get_prompt("reply_with_tool_results" if retry < 5 else "reply_with_tool_results2"), retry + 1)
        else:
            error_message = f"I can only handle a context length of {selected_llm['context_length']} tokens. The total length of all messages possibly exceeded my limit. Please start a new conversation, or cut parts of the chat history, or continue with other LLMs such as Gemini, which can process up to 1 million tokens."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
    except Exception:
        st.warning(f"An error occurred:\n{traceback.format_exc()}")


def style_message_content(content):
    if matched := re.search(r"^<think>([\s\S]*)</think>\s*([\s\S]*)", content):
        if reasoning := RE_COMPRESS_NEWLINES.sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{reasoning}</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<doc_content>([\s\S]*)</doc_content>\s*([\s\S]*)", content):
        if doc_content := RE_COMPRESS_NEWLINES.sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{doc_content[:300]}...</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<tool_results>([\s\S]*)</tool_results>\s*([\s\S]*)", content):
        return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{matched.group(1)[:300]}...</div>{matched.group(2)}"
    return content


def display_messages(index=0):
    messages = st.session_state["messages"][index:]
    for i, message in enumerate(messages, start=index):
        message_column, button_column = st.columns([74, 1])
        with message_column:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"User:\n{style_message_content(message['content'])}", unsafe_allow_html=True)
                    if "image_paths" in message:
                        if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                            st.image(image_paths, width=240)
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(f"AI ({st.session_state['llm']}):\n{style_message_content(message['content'])}", unsafe_allow_html=True)
        with button_column:
            if st.button("¯", key=f"delete_message_{i}", help="Delete this message", type="tertiary"):
                st.session_state["messages"].pop(i)
                st.session_state["revised_messages"] = sync_messages()
                st.rerun()
            st.download_button("ˇ", f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}", f"Message {i + 1} {now_and_choices()}.txt", "text/plain", key=f"download_message_{i}", help="Download this message", type="tertiary")
    return len(st.session_state["messages"])


@st.dialog("revise chat history", width="large")
def revise_messages():
    st.text_area("revise messages", height=window_height - 240, key="revised_messages", label_visibility="collapsed")
    if st.button("Update", type="tertiary", use_container_width=True):
        st.session_state["messages"] = update_messages()
        st.rerun()


chat_input = st.chat_input("Input a message", accept_file="multiple", file_type=["pdf", "csv", "xlsx", "xls", "jpg", "jpeg", "png"])
user_message = chat_input.text.strip() if chat_input else None
files = chat_input.files if chat_input else []
file_name_and_path_tuples = get_file_name_and_path_tuples([file for file in files if file.name not in {file_name for message in st.session_state["messages"] if "file_names" in message for file_name in message["file_names"]}])

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 18px;'><a href='https://friendsofexcellence.ai' style='text-decoration: none; color: inherit;'>优秀的朋友用的 AI for Friends of Excellence</a></h1>", unsafe_allow_html=True)

    with st.container(height=(window_height - 418) // 2, border=False):
        job_options = list(jobs.keys())
        captions = [jobs[job_option]["intro"] for job_option in job_options]
        st.radio("Job Options", job_options, key="job", on_change=select_job, captions=captions, label_visibility="collapsed")

    with st.container(height=(window_height - 536) // 2, border=False):
        llm_options = jobs[st.session_state["job"]]["llms"]
        st.pills("LLM Options", llm_options, key="llm", label_visibility="collapsed")

    with st.expander("More Settings", expanded=False):
        st.slider("Temperature", 0.0, 1.0, key="temperature", on_change=set_temperature)
        st.slider("Top P", 0.01, 1.0, key="top_p", on_change=set_top_p)

        answer_styles = ["Concise Conclusive", "Detailed Explanatory", "Formal Written"]
        st.pills("Answer Style", answer_styles, key="answer_style")

        answer_languages = ["简体中文", "English (American)", "English (British)"]
        st.pills("Answer Language", answer_languages, key="answer_language")

        pdf_types = ["Plain Text", "With Visuals"]
        st.pills("PDF Type", pdf_types, key="pdf_type")

    if user_message_audio := st.audio_input("Click the mic and speak", label_visibility="collapsed"):
        user_message = transcribe_user_message(user_message_audio)

    export_column, revise_column = st.columns([1, 1])
    with export_column:
        st.download_button("Export", st.session_state["revised_messages"], f"Messages {now_and_choices()}.txt", "text/plain", help="Export chat history", type="tertiary", use_container_width=True)
    with revise_column:
        if st.button("Revise", help="Revise chat history", type="tertiary", use_container_width=True):
            revise_messages()

if user_message:
    append_user_message(user_message, file_name_and_path_tuples)
    displayed_quantity = display_messages()
    chat()
    display_messages(displayed_quantity)
else:
    display_messages()
