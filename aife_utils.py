import streamlit as st
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
import mmap
import re
import regex
from concurrent.futures import ThreadPoolExecutor, wait
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from PIL import Image
import fitz
import codecs
from charset_normalizer import detect
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from random import choices
from string import digits, ascii_lowercase
import dateparser
import fcntl
import time
import threading

for directory in ["temp-data", "temp-images", "uploaded-files"]:
    os.makedirs(directory, exist_ok=True)

tenant_id = st.secrets["tenant_id"]
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
vault_url = st.secrets["vault_url"]

client = SecretClient(vault_url=vault_url, credential=ClientSecretCredential(tenant_id, client_id, client_secret))

def retrieve(secret_name):
    return client.get_secret(secret_name).value

YUSISTORAGE_CONNECTION_STRING = retrieve("YusiStorageConnectionString")
SMTP_PASSWORD = retrieve("SmtpPassword")

RE_NORMALIZE_NEWLINES = re.compile(r"\r\n?")
RE_REMOVE_MARKDOWN_COMPOSITE_LINKS = re.compile(r"\s*[!@#]?\[(?:[^\[\]]*\[[^\]]*\][^\[\]]*|[^\[\]]*)\]\([^)]*\)")
RE_REMOVE_MARKDOWN_BASIC_LINKS = re.compile(r"\s*\[[^\[\]]*\]\([^)]*\)")
RE_REMOVE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_REMOVE_INVALID_LINES = regex.compile(r'^[^\p{Letter}\p{Number}\[\]\(\)]*$', flags=regex.MULTILINE)
RE_COMPRESS_NEWLINES = re.compile(r"\n+")
REMOVE_FILETYPE = re.compile(r"\bfiletype:\S+(?:.*\bfiletype:\S+)?")

weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekdays_zh = ["一", "二", "三", "四", "五", "六", "日"]
months_en = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def manage_futures(requests, max_concurr=20):
    with ThreadPoolExecutor(min(len(requests), max_concurr)) as executor:
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
                print(f"Error in {function.__name__} with arguments {arguments}: {e}")
        return results


async def manage_futures_async(requests, max_concurr=20):
    semaphore = asyncio.Semaphore(max_concurr)
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
            print(f"Error in {function.__name__} with arguments {arguments}: {e}")
    return results


def upload_to_container(file_path):
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    for attempt in range(3):
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
                block_ids = [f"{i + 1:06d}" for i in range(file_size // 524288 + (file_size % 524288 > 0))]
                requests = [(upload_block, (block_id, mm[i * 524288:(i + 1) * 524288])) for i, block_id in enumerate(block_ids)]
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


def ensure_utf8_csv(table_path):
    try:
        if table_path.endswith(".csv"):
            with open(table_path, "rb") as f:
                encoding = codecs.lookup(detect(f.read(min(32768, os.path.getsize(table_path))))["encoding"]).name
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
        print(f"Error in ensure_utf8_csv: {e}")
    return None


def ensure_utf8_csvs(table_paths):
    requests = []
    for table_path in table_paths:
        if table_path:
            request = (ensure_utf8_csv, (table_path,))
            requests.append(request)
    return [result for result, function, arguments in manage_futures(requests) if result]


def extract_text_with_xml_tag(text, tag):
    if matched := re.search(rf"<{tag}>([\s\S]*)</{tag}>", text):
        return matched.group(1).strip()
    else:
        return re.sub(r"<[^>]+>", "", text).strip()


def extract_text_with_tags(text, tags):
    result = {}
    for tag in tags:
        pattern = r"(?i)" + rf"{tag}:\s*([\s\S]*?)(?=\s*(?:{'|'.join(tag + ':' for tag in tags)}|$))"
        if match := re.search(pattern, text):
            content = match.group(1).strip()
            result[tag] = content
        else:
            return None
    return result


def extract_text_with_pattern(text, pattern):
    if markers := [(marker.start(), marker.group()) for marker in re.finditer(pattern, text)]:
        return {i + 1: text[marker[0] + len(marker[1]):markers[i + 1][0] if i < len(markers) - 1 else len(text)].strip() for i, marker in enumerate(markers)}
    else:
        return None


def filter_words(text, words):
    for word in words:
        text = text.replace(word, "")
    return text


def get_curr_time():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def now_and_choices():
    return get_curr_time().strftime("%Y%m%d%H%M%S") + "".join(choices(digits + ascii_lowercase, k=4))


def hours_ago(hours):
    return get_curr_time() - timedelta(hours=hours)


def hours_later(hours):
    return get_curr_time() + timedelta(hours=hours)


def days_ago(days):
    return get_curr_time() - timedelta(days=days)


def days_later(days):
    return get_curr_time() + timedelta(days=days)


def get_day_suffix(day):
    return "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def get_curr_day_en():
    return f"{get_curr_time().day}{get_day_suffix(get_curr_time().day)}"


def get_curr_day_zh():
    return f"{get_curr_time().day}日"


def get_prev_day_en():
    return f"{(get_curr_time() - timedelta(days=1)).day}{get_day_suffix((get_curr_time() - timedelta(days=1)).day)}"


def get_prev_day_zh():
    return f"{(get_curr_time() - timedelta(days=1)).day}日"


def get_next_day_en():
    return f"{(get_curr_time() + timedelta(days=1)).day}{get_day_suffix((get_curr_time() + timedelta(days=1)).day)}"


def get_next_day_zh():
    return f"{(get_curr_time() + timedelta(days=1)).day}日"


def get_curr_weekday_en():
    return weekdays_en[get_curr_time().weekday()]


def get_curr_weekday_zh():
    return f"星期{weekdays_zh[get_curr_time().weekday()]}"


def get_prev_weekday_en():
    return weekdays_en[(get_curr_time().weekday() - 1) % 7]


def get_prev_weekday_zh():
    return f"星期{weekdays_zh[(get_curr_time().weekday() - 1) % 7]}"


def get_next_weekday_en():
    return weekdays_en[(get_curr_time().weekday() + 1) % 7]


def get_next_weekday_zh():
    return f"星期{weekdays_zh[(get_curr_time().weekday() + 1) % 7]}"


def get_curr_month_en():
    return months_en[get_curr_time().month - 1]


def get_curr_month_zh():
    return f"{get_curr_time().month}月"


def get_prev_month_en():
    return months_en[(get_curr_time().month - 2) % 12]


def get_prev_month_zh():
    return f"{(get_curr_time().month - 1) or 12}月"


def get_next_month_en():
    return months_en[get_curr_time().month % 12]


def get_next_month_zh():
    return f"{(get_curr_time().month % 12) + 1}月"


def get_curr_year():
    return f"{get_curr_time().year}"


def get_prev_year():
    return f"{get_curr_time().year - 1}"


def get_next_year():
    return f"{get_curr_time().year + 1}"


def get_today_with_weekday_en():
    curr_time = get_curr_time()
    return f"{months_en[curr_time.month - 1]} {curr_time.day}{get_day_suffix(curr_time.day)}, {weekdays_en[curr_time.weekday()]}"


def get_today_with_weekday_zh():
    curr_time = get_curr_time()
    return f"{curr_time.month}月{curr_time.day}日，星期{weekdays_zh[curr_time.weekday()]}"


def get_yesterday_with_weekday_en():
    yesterday = get_curr_time() - timedelta(days=1)
    return f"{months_en[yesterday.month - 1]} {yesterday.day}{get_day_suffix(yesterday.day)}, {weekdays_en[yesterday.weekday()]}"


def get_yesterday_with_weekday_zh():
    yesterday = get_curr_time() - timedelta(days=1)
    return f"{yesterday.month}月{yesterday.day}日，星期{weekdays_zh[yesterday.weekday()]}"


def get_tomorrow_with_weekday_en():
    tomorrow = get_curr_time() + timedelta(days=1)
    return f"{months_en[tomorrow.month - 1]} {tomorrow.day}{get_day_suffix(tomorrow.day)}, {weekdays_en[tomorrow.weekday()]}"


def get_tomorrow_with_weekday_zh():
    tomorrow = get_curr_time() + timedelta(days=1)
    return f"{tomorrow.month}月{tomorrow.day}日，星期{weekdays_zh[tomorrow.weekday()]}"


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


def year_start(year):
    return (datetime(year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai")) if year <= get_curr_time().year else datetime(get_curr_time().year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai")))


def year_end(year):
    return (datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=ZoneInfo("Asia/Shanghai")) if year < get_curr_time().year else get_curr_time())


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
        time.sleep(max(0, (next_run - get_curr_time()).total_seconds()))  # 每次循环都需要最新时间
        run_with_lock(function)
        next_run += timedelta(days=1)


cleanup_thread = threading.Thread(target=scheduled_run, args=(10, 0, clean_yesterday_files), daemon=True)
cleanup_thread.start()
