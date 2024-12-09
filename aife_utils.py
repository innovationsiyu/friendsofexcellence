import streamlit as st
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import os
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from pathlib import Path
from aife_time import now, hours_ago

for directory in ["temp-data", "temp-images", "uploaded-files"]:
    os.makedirs(directory, exist_ok=True)

tenant_id = st.secrets["tenant_id"]
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
vault_url = st.secrets["vault_url"]

credential = ClientSecretCredential(tenant_id, client_id, client_secret)
client = SecretClient(vault_url=vault_url, credential=credential)


def retrieve(secret_name):
    return client.get_secret(secret_name).value


YUSISTORAGE_CONNECTION_STRING = retrieve("YusiStorageConnectionString")

SMTP_PASSWORD = retrieve("SmtpPassword")


def manage_thread(requests, thread_count=20):
    if requests:
        with ThreadPoolExecutor(min(len(requests), thread_count)) as executor:
            futures = [(executor.submit(function, *arguments), function, arguments) for function, *arguments in requests]
            return [(future.result(), function.__name__, arguments) for future, function, arguments in futures]
    return []


def upload_to_container(file_path):
    for attempt in range(3):
        try:
            blob_client = BlobServiceClient.from_connection_string(YUSISTORAGE_CONNECTION_STRING).get_blob_client("temp-data", os.path.basename(file_path))
            chunk_size = 1 * 1024 * 1024
            block_ids = []
            with open(file_path, "rb") as file:
                chunks = iter(lambda: file.read(chunk_size), file.read(0))
                for i, chunk in enumerate(chunks, 1):
                    block_id = f"{i:06d}"
                    block_ids.append(block_id)
                    blob_client.stage_block(block_id=block_id, data=chunk)
                    print(f"Block {i} uploaded successfully")
                blob_client.commit_block_list(block_ids)
                print("File uploaded successfully")
                return blob_client.url
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to upload file after maximum retries")
    return None


def create_and_send_email(file_path, file_url, to_email):
    smtp_server = "smtp.exmail.qq.com"
    smtp_port = 465
    smtp_user = "siyu@yusiconsulting.com"
    filename = os.path.splitext(os.path.basename(file_path))[0]
    body = f"""这里是优秀的朋友用的 AI for Friends of Excellence.<br>
<br>
您请求的文件已经生成。请在24小时内点击下方文件名下载：<br>
The requested file has been generated. Please click the filename below to download it within 24 hours:<br>
<br>
<a href="{file_url}">{filename}</a><br>
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
    msg["Date"] = now().strftime("%a, %d %b %Y %H:%M:%S +0800")
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


def upload_and_send(file_path, to_email=None):
    file_url = upload_to_container(file_path)
    if file_url:
        if to_email:
            create_and_send_email(file_path, file_url, to_email)
        return file_url
    return None


def del_temp_blobs(cutoff_time):
    try:
        blob_service = BlobServiceClient.from_connection_string(YUSISTORAGE_CONNECTION_STRING)
        for container in ["temp-data", "temp-images"]:
            container_client = blob_service.get_container_client(container)
            deleted_blobs = [container_client.get_blob_client(blob).delete_blob() for blob in container_client.list_blobs() if blob.last_modified.timestamp() < cutoff_time]
            print(f"Deleted {len(deleted_blobs)} blobs from {container} container")
    except Exception as e:
        print(f"Error in del_temp_blobs: {e}")


def del_temp_files(cutoff_time):
    try:
        for directory in ["temp-data", "temp-images", "uploaded-files"]:
            deleted_files = [file.unlink() for file in Path(directory).iterdir() if file.stat().st_ctime < cutoff_time]
            print(f"Deleted {len(deleted_files)} files from {directory} directory")
    except Exception as e:
        print(f"Error in del_temp_files: {e}")


def clean_yesterday_files():
    cutoff_time = hours_ago(24).timestamp()
    del_temp_blobs(cutoff_time)
    del_temp_files(cutoff_time)
