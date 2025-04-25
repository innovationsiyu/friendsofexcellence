import streamlit as st
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
import mmap
import re
from concurrent.futures import ThreadPoolExecutor, wait
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from PIL import Image
import fitz
import ast
import codecs
from charset_normalizer import detect
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from random import choices
from string import digits, ascii_lowercase, ascii_uppercase
import dateparser
import fcntl
import time
import threading


for directory in ["temp-data", "temp-images", "uploaded-files", ".streamlit"]:
    os.makedirs(directory, exist_ok=True)

with open(".streamlit/config.toml", "w") as f:
    f.write("""[theme]
base="light"
primaryColor="#283692"
baseRadius="none"
baseFontSize=15""")

tenant_id = st.secrets["tenant_id"]
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
vault_url = st.secrets["vault_url"]

client = SecretClient(vault_url=vault_url, credential=ClientSecretCredential(tenant_id, client_id, client_secret))

def retrieve(secret_name):
    return client.get_secret(secret_name).value

YUSISTORAGE_CONNECTION_STRING = retrieve("YusiStorageConnectionString")
SMTP_PASSWORD = retrieve("SmtpPassword")

weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekdays_zh = ["一", "二", "三", "四", "五", "六", "日"]
months_en = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def manage_futures(requests, max_concurrent=20):
    with ThreadPoolExecutor(min(len(requests), max_concurrent)) as executor:
        futures = []
        for function, arguments in requests:
            if isinstance(arguments, tuple):
                future = executor.submit(function, *arguments)
            elif isinstance(arguments, dict):
                future = executor.submit(function, **arguments)
            else:
                raise TypeError(f"Arguments must be a tuple or dict, got {type(arguments)}")
            futures.append(future)
        wait(futures)
        results = []
        for i, future in enumerate(futures):
            function, arguments = requests[i]
            try:
                result = future.result()
                results.append((result, function, arguments))
            except Exception as e:
                results.append((None, function, arguments))
                print(f"Error in {function.__name__} with arguments {arguments}: {e}")
        return results


async def manage_futures_async(requests, max_concurrent=20):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def semaphore_task(function, arguments):
        async with semaphore:
            if isinstance(arguments, tuple):
                return await function(*arguments)
            elif isinstance(arguments, dict):
                return await function(**arguments)
            else:
                raise TypeError(f"Arguments must be a tuple or dict, got {type(arguments)}")
    futures = []
    for function, arguments in requests:
        future = asyncio.create_task(semaphore_task(function, arguments))
        futures.append(future)
    await asyncio.gather(*futures)
    results = []
    for i, future in enumerate(futures):
        function, arguments = requests[i]
        try:
            result = future.result()
            results.append((result, function, arguments))
        except Exception as e:
            results.append((None, function, arguments))
            print(f"Error in {function.__name__} with arguments {arguments}: {e}")
    return results


def upload_to_container(file_path):
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    block_size = 262144
    for attempt in range(2):
        try:
            blob_client = BlobServiceClient.from_connection_string(YUSISTORAGE_CONNECTION_STRING).get_blob_client("temp-data", file_name)
            def upload_block(block_id, data):
                for attempt in range(3):
                    try:
                        blob_client.stage_block(block_id=block_id, data=data)
                        print(f"Block {block_id} uploaded successfully")
                        return True
                    except Exception:
                        continue
                return False
            with open(file_path, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                block_ids = [f"{i + 1:06d}" for i in range(file_size // block_size + (file_size % block_size > 0))]
                requests = [(upload_block, (block_id, mm[i * block_size:(i + 1) * block_size])) for i, block_id in enumerate(block_ids)]
                if all(result[0] for result in manage_futures(requests)):
                    blob_client.commit_block_list(block_ids)
                    print("File uploaded successfully")
                    return blob_client.url
        except Exception:
            continue
    return None


def email_file_url(file_path, file_url, to_email):
    smtp_server = "smtp.exmail.qq.com"
    smtp_port = 465
    smtp_user = "siyu@yusiconsulting.com"
    body = f"""这里是优秀的朋友用的 AI for Friends of Excellence.<br>
<br>
您请求的文件已经生成。请在24小时内点击下载：<br>
The requested file has been generated. Please click to download it within 24 hours:<br>
<br>
<a href="{file_url}">{file_path}</a><br>
<br>
如果您想咨询专属服务，请回复此邮件联系思宇，或者加我微信"innovationsiyu"。<br>
If you would like to enquire about exclusive services, please reply to this email to contact Siyu, or add me on WeChat at "innovationsiyu".<br>
<br>
<br>
<a href="https://friendsofexcellence.ai">Friends of Excellence.ai</a>"""
    msg = MIMEMultipart()
    msg["From"] = formataddr(("Siyu", smtp_user))
    msg["To"] = to_email
    msg["Subject"] = "Your requested file is ready - Friends of Excellence.ai"
    msg["Date"] = get_curr_time().strftime("%a, %d %b %Y %H:%M:%S +0800")
    msg.attach(MIMEText(body, "html"))
    for attempt in range(3):
        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_user, SMTP_PASSWORD)
                server.sendmail(smtp_user, [to_email], msg.as_string())
            print("Email sent successfully")
            return
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to send email after maximum retries")


def upload_and_email_file(file_path, to_email=None):
    if file_url := upload_to_container(file_path):
        if to_email:
            email_file_url(file_path, file_url, to_email)
        return file_url
    return None


def resize_image(image_path, side_limit, width_specific):
    with Image.open(image_path) as f:
        if width_specific and f.size[0] > side_limit:
            ratio = side_limit / f.size[0]
        elif not width_specific and max(f.size) > side_limit:
            ratio = side_limit / max(f.size)
        else:
            return
        resized = f.resize((int(f.size[0] * ratio), int(f.size[1] * ratio)), Image.Resampling.LANCZOS)
        with open(image_path, "wb") as out_f:
            resized.save(out_f, f.format)


def resize_images(image_paths, side_limit, width_specific):
    requests = []
    for image_path in image_paths:
        if image_path:
            request = (resize_image, (image_path, side_limit, width_specific))
            requests.append(request)
    manage_futures(requests)


def pdf_to_images(pdf_path):
    image_paths = []
    with fitz.open(pdf_path) as f:
        for i in range(f.page_count):
            image_path = os.path.join("temp-images", f"{now_and_choices()}.png")
            f.load_page(i).get_pixmap().save(image_path)
            if os.path.isfile(image_path):
                image_paths.append(image_path)
            else:
                image_paths.append(None)
    return image_paths


def save_as_txt(text):
    txt_path = os.path.join("temp-data", f"{now_and_choices()}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return txt_path


def get_text(text_paths, to_txt=False):
    text_paths = text_paths if isinstance(text_paths, list) else [text_paths]
    texts = []
    for text_path in text_paths:
        with open(text_path, "r", encoding="utf-8") as f:
            if text := f.read().strip():
                try:
                    text_chunks = ast.literal_eval(text)
                    if isinstance(text_chunks, list) and all(isinstance(text_chunk, str) for text_chunk in text_chunks):
                        texts.extend(text_chunks)
                    else:
                        texts.append(text)
                except Exception:
                    texts.append(text)
    if to_txt:
        txt_path = save_as_txt(str(texts))
        return texts, txt_path
    return texts


def ensure_utf_8_csv(table_path):
    try:
        if table_path.endswith(".csv"):
            with open(table_path, "rb") as f:
                if detected := detect(f.read(min(32768, os.path.getsize(table_path)))).get("encoding"):
                    encoding = codecs.lookup(detected).name
                else:
                    encoding = "utf-8"
            df = pd.read_csv(table_path, encoding=encoding)
        elif table_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(table_path, engine="openpyxl")
        else:
            return None
        if first_valid_column := next((column for column in df.columns if pd.notna(column) or df[column].notna().any()), None):
            df[first_valid_column] = df[first_valid_column].ffill()
            csv_path = os.path.splitext(table_path)[0] + ".csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            return csv_path
    except Exception as e:
        print(f"Error in ensure_utf_8_csv: {e}")
    return None


def ensure_utf_8_csvs(table_paths):
    requests = []
    for table_path in table_paths:
        if table_path:
            request = (ensure_utf_8_csv, (table_path,))
            requests.append(request)
    return [result for result, function, arguments in manage_futures(requests) if result]


def extract_text_with_xml_tags(text, tags):
    tags = tags if isinstance(tags, list) else [tags]
    results = []
    for tag in tags:
        if matched := re.search(rf"<{tag}>(?P<result>[\s\S]*)</{tag}>", text):
            results.append(matched.group("result").strip())
        else:
            results.append(re.sub(r"<[^>]+>", "\n", text).strip())
    return tuple(results) if len(tags) > 1 else results[0]


def extract_text_with_tags(text, tags):
    tags = tags if isinstance(tags, list) else [tags]
    result = []
    for tag in tags:
        pattern = r"(?i)" + rf"{tag}\s*(?P<result>[\s\S]*?)(?=\s*(?:{'|'.join(tags)}|$))"
        if matched := re.search(pattern, text):
            result.append(matched.group("result").strip())
        else:
            result.append(None)
    return result


def extract_text_with_pattern(text, pattern):
    if matches := [(match.start(), match.end()) for match in re.finditer(pattern, text)]:
        result = []
        for start, end in list(zip([0] + [match[1] for match in matches], [match[0] for match in matches] + [len(text)]))[1:]:
            result.append(text[start:end].strip())
        return result
    return [text.strip()]


def filter_words(text, words):
    for word in words:
        text = text.replace(word, "")
    return text


def get_day_suffix(day):
    return "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def get_curr_user_time():
    return datetime.now(ZoneInfo(st.session_state["timezone"]))


def get_curr_hour_and_minute():
    curr_user_time = get_curr_user_time()
    return f"{curr_user_time.hour:02d}:{curr_user_time.minute:02d}"


def get_today_with_weekday_en():
    curr_user_time = get_curr_user_time()
    return f"{weekdays_en[curr_user_time.weekday()]}, {curr_user_time.day}{get_day_suffix(curr_user_time.day)} {months_en[curr_user_time.month - 1]} {curr_user_time.year}"


def get_today_with_weekday_zh():
    curr_user_time = get_curr_user_time()
    return f"{curr_user_time.year}年{curr_user_time.month}月{curr_user_time.day}日，星期{weekdays_zh[curr_user_time.weekday()]}"


def get_curr_time():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def now_and_choices():
    return get_curr_time().strftime("%Y%m%d%H%M%S") + "".join(choices(digits + ascii_lowercase + ascii_uppercase, k=6))


def get_weekdays_with_dates_en():
    curr_time = get_curr_time()
    week_start = curr_time - timedelta(days=curr_time.weekday())
    dates = [week_start + timedelta(days=i) for i in range(-7, 14)]
    return "; ".join([", ".join([f"{'this' if i == 7 else 'last' if i == 0 else 'next'} {weekdays_en[d.weekday()]} is {months_en[d.month - 1]} {d.day}{get_day_suffix(d.day)}" for d in dates[i:i + 7]]) for i in (7, 0, 14)])


def get_weekdays_with_dates_zh():
    curr_time = get_curr_time()
    week_start = curr_time - timedelta(days=curr_time.weekday())
    dates = [week_start + timedelta(days=i) for i in range(-7, 14)]
    return "；".join(["，".join([f"{'本' if i == 7 else '上' if i == 0 else '下'}周{weekdays_zh[d.weekday()]}是{d.month}月{d.day}日" for d in dates[i:i + 7]]) for i in (7, 0, 14)])


def get_recent_dates_iso(days):
    return [(get_curr_time() - timedelta(days=i)).date().isoformat() for i in range(days)]


def iso_date(timestamp):
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp, ZoneInfo("Asia/Shanghai"))
        else:
            dt = dateparser.parse(timestamp, settings={"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Asia/Shanghai"})
        return dt.date().isoformat()
    except Exception:
        return timestamp


def hours_ago(hours):
    return get_curr_time() - timedelta(hours=hours)


def hours_later(hours):
    return get_curr_time() + timedelta(hours=hours)


def days_ago(days):
    return get_curr_time() - timedelta(days=days)


def days_later(days):
    return get_curr_time() + timedelta(days=days)


def year_start(year):
    curr_year = get_curr_time().year
    if year <= curr_year:
        return datetime(year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        return datetime(curr_year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))


def year_end(year):
    curr_year = get_curr_time().year
    if year < curr_year:
        return datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=ZoneInfo("Asia/Shanghai"))
    else:
        return get_curr_time()


def delete_temp_blobs(cutoff_time):
    try:
        blob_service = BlobServiceClient.from_connection_string(YUSISTORAGE_CONNECTION_STRING)
        for container in ["temp-data", "temp-images"]:
            container_client = blob_service.get_container_client(container)
            deleted_blobs = [container_client.get_blob_client(blob).delete_blob() for blob in container_client.list_blobs() if blob.last_modified.timestamp() < cutoff_time]
            print(f"Deleted {len(deleted_blobs)} blobs from {container} container")
    except Exception as e:
        print(f"Error in delete_temp_blobs: {e}")


def delete_temp_files(cutoff_time):
    try:
        for directory in ["temp-data", "temp-images", "uploaded-files"]:
            deleted_files = [os.remove(file_path) for file_path in Path(directory).iterdir() if file_path.stat().st_ctime < cutoff_time]
            print(f"Deleted {len(deleted_files)} files from {directory} directory")
    except Exception as e:
        print(f"Error in delete_temp_files: {e}")


def clean_yesterday_files():
    cutoff_time = hours_ago(24).timestamp()
    delete_temp_blobs(cutoff_time)
    delete_temp_files(cutoff_time)


def run_with_lock(function):
    LOCK = "cleanup.lock"
    try:
        with open(LOCK, "w") as lock:
            fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                function()
            finally:
                fcntl.lockf(lock, fcntl.LOCK_UN)
    except Exception:
        pass


def scheduled_run(start_hour, start_minute, function):
    curr_time = get_curr_time()
    next_run = curr_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    if curr_time > next_run:
        next_run += timedelta(days=1)
    while True:
        time.sleep(max(0, (next_run - get_curr_time()).total_seconds()))
        run_with_lock(function)
        next_run += timedelta(days=1)


cleanup_thread = threading.Thread(target=scheduled_run, args=(10, 0, clean_yesterday_files), daemon=True)
cleanup_thread.start()
