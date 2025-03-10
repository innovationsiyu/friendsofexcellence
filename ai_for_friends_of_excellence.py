import streamlit as st
import os
import re
import hashlib
import tempfile
from pathvalidate import sanitize_filename
import regex
import copy
import base64
import traceback
from aife_tools import get_prompt, get_tools, route_llm_request, speech_to_text, jobs, llms, parse_pdfs
from aife_utils import RE_COMPRESS_NEWLINES, resize_images, ensure_utf8_csvs, now_and_choices

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.components.v1.html(
    """
    <script>
    const updateWindowHeight = () => {
        const windowHeight = window.parent.innerHeight;
        pdf.cookie = `window_height=${windowHeight};path=/`;
    };
    updateWindowHeight();
    window.parent.addEventListener('resize', updateWindowHeight);
    </script>
    """,
    height=0
)

window_height = int(st.context.cookies.get("window_height", 800))

st.session_state["job"] = st.query_params.get("job", st.session_state.get("job", "Text chat"))
st.session_state["temperature"] = float(st.query_params.get("temperature", st.session_state.get("temperature", 0.5)))
st.session_state["llm"] = st.session_state.get("llm", jobs[st.session_state["job"]]["llms"][0])
st.session_state["chat_history"] = st.session_state.get("chat_history", [])
st.session_state["chat_history_editable"] = st.session_state.get("chat_history_editable")
st.session_state["is_chat_history_edited"] = st.session_state.get("is_chat_history_edited", False)
st.session_state["pdf_type"] = st.session_state.get("pdf_type", "Plain text")
st.session_state["edit_mode"] = st.session_state.get("edit_mode", False)
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
        table_paths = ensure_utf8_csvs(table_paths)
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


def sync_chat_history():
    pattern = r"(User:\n|AI \([^)]+\):\n)"
    segments = ["User:\n"] + [segment for segment in re.split(pattern, st.session_state["chat_history_editable"]) if segment.strip()]
    return [{"role": "user" if segments[i] == "User:\n" else "assistant", "content": segments[i + 1].strip()} for i in range(len(segments) - 1) if re.match(pattern, segments[i]) and not re.match(pattern, segments[i + 1])]


def sync_chat_history_editable():
    return "\n\n".join([f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}" for message in st.session_state["chat_history"]])


def is_chat_history_edited():
    st.session_state["is_chat_history_edited"] = True


def interleave_messages(messages):
    for i in reversed(range(1, len(messages))):
        last_message, current_message = messages[i - 1], messages[i]
        if last_message["role"] == current_message["role"] and isinstance(current_message["content"], str):
            last_message["content"] += "\n\n" + current_message.pop("content")
            messages.pop(i)
    return messages


def is_in_chinese(text):
    return bool(regex.search(r"\p{Han}", text))


def chat(continue_message=None, retry=0):
    selected_job = jobs[st.session_state["job"]]
    messages = copy.deepcopy(st.session_state["chat_history"])
    in_chinese = is_in_chinese(messages[0]["content"])
    for i, message in enumerate(messages):
        if "image_paths" in message:
            if image_paths := [image_path for image_path in message["image_paths"] if os.path.isfile(image_path)]:
                messages[i]["content"] = [{"type": "text", "text": message["content"]}, *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"}} for image_path in image_paths]]
    if continue_message:
        messages.append({"role": "user", "content": continue_message})
    messages = [{"role": "system", "content": get_prompt(selected_job["system_message"])}] + interleave_messages(messages)
    selected_llm = llms[st.session_state["llm"] if st.session_state["llm"] else jobs[st.session_state["job"]]["llms"][0]]
    llm_result = route_llm_request(selected_llm["configs"], messages, st.session_state["temperature"], get_tools(selected_job["tools"]))
    if isinstance(llm_result, str):
        st.session_state["chat_history"].append({"role": "assistant", "content": llm_result})
        st.session_state["chat_history_editable"] = sync_chat_history_editable()
    elif isinstance(llm_result, dict) and llm_result["type"] == "tool_results":
        st.session_state["chat_history"].append({"role": "assistant", "content": llm_result["tool_results"]})
        chat(get_prompt("reply_with_tool_results" if retry < 5 else "reply_with_tool_results2", in_chinese=in_chinese), retry + 1)
    else:
        error_message = (f"我只能处理{selected_llm["context_length"]}个tokens的上下文长度。当前全部消息总长度可能超过了极限。请开启新会话，或者剪切部分历史消息，或者选择其它大语言模型继续对话，例如MiniMax，可以处理多达100万个tokens。" if in_chinese else f"I can only handle a context length of {selected_llm["context_length"]} tokens. The total length of all messages possibly exceeded my limit. Please start a new conversation thread, or cut parts of the chat history, or continue with other LLMs such as Gemini, which can process up to 1 million tokens.")
        st.session_state["chat_history"].append({"role": "assistant", "content": error_message})


def style_message_content(content):
    if matched := re.search(r"^<think>([\s\S]*)</think>\s*([\s\S]*)", content):
        if reasoning_content := RE_COMPRESS_NEWLINES.sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{reasoning_content}</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<doc_content>([\s\S]*)</doc_content>\s*([\s\S]*)", content):
        if doc_content := RE_COMPRESS_NEWLINES.sub("\n", matched.group(1).strip()):
            return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{doc_content[:300]}...</div>{matched.group(2)}"
        return matched.group(2)
    elif matched := re.search(r"^<tool_results>([\s\S]*)</tool_results>\s*([\s\S]*)", content):
        return f"<div style='font-size: 14px; color: #B0B0B0; white-space: pre-wrap;'>{matched.group(1)[:300]}...</div>{matched.group(2)}"
    return content


def show_messages():
    messages = st.session_state["chat_history"]
    for i, message in enumerate(messages):
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
                messages.pop(i)
                st.session_state["chat_history_editable"] = sync_chat_history_editable()
                st.rerun()
            st.download_button("ˇ", f"User:\n{message['content']}" if message["role"] == "user" else f"AI ({st.session_state['llm']}):\n{message['content']}", f"Message {i + 1} {now_and_choices()}.txt", "text/plain", key=f"download_message_{i}", help="Download this message", type="tertiary")


chat_input = st.chat_input("Input a message", accept_file=True, file_type=["pdf", "csv", "xlsx", "xls", "jpg", "jpeg", "png"])
user_message = chat_input.text.strip() if chat_input else None
files = chat_input.files if chat_input else []
new_files = [file for file in files if file.name not in {file_name for message in st.session_state["chat_history"] if "file_names" in message for file_name in message["file_names"]}]
file_name_and_path_tuples = get_file_name_and_path_tuples(new_files)

with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 18px;'><a href='https://friendsofexcellence.ai' style='text-decoration: none; color: inherit;'>优秀的朋友用的 AI for Friends of Excellence</a></h1>", unsafe_allow_html=True)

    with st.container(height=(window_height - 346) // 2, border=False):
        job_options = list(jobs.keys())
        captions = [jobs[job_option]["intro"] for job_option in job_options]
        st.radio("Job options", job_options, key="job", on_change=select_job, captions=captions, label_visibility="collapsed")

    st.number_input("Set temperature", 0.0, 1.0, step=0.1, key="temperature", on_change=set_temperature)

    with st.container(height=(window_height - 508) // 2, border=False):
        llm_options = jobs[st.session_state["job"]]["llms"]
        st.pills("LLM options", llm_options, key="llm", label_visibility="collapsed")

    with st.expander("Audios and settings", expanded=False):
        if user_message_audio := st.audio_input("Click the mic and speak"):
            buffer = user_message_audio.getbuffer()
            current_audio_hash = hashlib.md5(buffer[:min(len(buffer), 882000)]).hexdigest()

            if current_audio_hash != st.session_state["last_audio_hash"]:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(buffer)
                try:
                    user_message = speech_to_text(f.name)
                except Exception:
                    st.warning(f"An error occurred:\n{traceback.format_exc()}")
                finally:
                    st.session_state["last_audio_hash"] = current_audio_hash
                    os.remove(f.name)

        pdf_types = ["Plain text", "With visuals"]
        st.pills("Chose the type of PDF", pdf_types, key="pdf_type")

    st.toggle("Edit mode", key="edit_mode")

if user_message:
    if st.session_state["is_chat_history_edited"]:
        st.session_state["chat_history"] = sync_chat_history()
        st.session_state["is_chat_history_edited"] = False

    try:
        if file_name_and_path_tuples:
            pdf_names, pdf_paths = get_pdfs(file_name_and_path_tuples)
            if pdf_names and pdf_paths:
                pdf_contents, txt_path = parse_pdfs(pdf_paths, st.session_state["pdf_type"])
                st.session_state["chat_history"].append({"role": "user", "content": f"<doc_content>{'\n\n'.join(pdf_contents[key] for key in sorted(pdf_contents))}</doc_content>\n\nThe path of the TXT file containing the doc_content: {txt_path}", "file_names": pdf_names})
            table_names, table_paths = get_tables(file_name_and_path_tuples)
            if table_names and table_paths:
                st.session_state["chat_history"].append({"role": "user", "content": f"The paths of the CSV files: {table_paths}", "file_names": table_names})
            image_names, image_paths = get_images(file_name_and_path_tuples)
            if image_names and image_paths and jobs[st.session_state["job"]]["category"] == "multimodal":
                st.session_state["chat_history"].append({"role": "user", "content": user_message, "file_names": image_names, "image_paths": image_paths})
            else:
                st.session_state["chat_history"].append({"role": "user", "content": user_message})
        else:
            st.session_state["chat_history"].append({"role": "user", "content": user_message})
        chat()
    except Exception:
        st.warning(f"An error occurred:\n{traceback.format_exc()}")

if st.session_state["edit_mode"]:
    st.text_area("Edit messages", height=window_height-310, key="chat_history_editable", on_change=is_chat_history_edited, label_visibility="collapsed")
else:
    show_messages()

st.components.v1.html(
    """
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        let lastContent = '';
        function getLast30Characters(string) { return string.slice(-30); }
        function checkAndScroll() {
            const textArea = window.parent.document.querySelector('textarea[aria-label="Edit messages"]');
            if (textArea) {
                const textAreaValue = textArea.value;
                if (getLast30Characters(textAreaValue) !== getLast30Characters(lastContent)) {
                    textArea.scrollTop = textArea.scrollHeight;
                    lastContent = textAreaValue;
                }
            }
        }
        setTimeout(() => {
            const textArea = window.parent.document.querySelector('textarea[aria-label="Edit messages"]');
            if (textArea) textArea.scrollTop = textArea.scrollHeight;
        }, 300);
        setInterval(checkAndScroll, 500);
    });
    </script>
    """,
    height=0
)
