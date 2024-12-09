import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from aife_utils import retrieve, manage_thread

SPIDER_API_KEY = retrieve("Spider")


def purify(text):
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'\s*[!@#]?\[(?:[^\[\]]*\[[^\]]*\][^\[\]]*|[^\[\]]*)\]\([^)]*\)', '', text)
    text = re.sub(r'\s*\[[^\[\]]*\]\([^)]*\)', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^[^A-Za-z0-9\u0080-\uffff]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '\n', text)
    return text[:30000].strip()


def reader(target_url, delay=1):
    url = f"https://r.jina.ai/{target_url}"
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, timeout=20)
            if response.text:
                return purify(response.text)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Failed to get a valid response after maximum retries")
    return None


def spider(target_url, delay=1):
    url = "https://api.spider.cloud/crawl"
    headers = {
        "Authorization": f"Bearer {SPIDER_API_KEY}",
    }
    data = {
        "url": target_url,
        "limit": 1,
        "return_format": "markdown",
    }
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=20).json()
            content = response[0].get("content")
            if content:
                return purify(content)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Failed to get a valid response after maximum retries")
    return None


def get_web_text(target_url):
    for index, request in enumerate([reader, spider]):
        try:
            text = request(target_url)
            if text:
                if len(text) >= 500 or index == len([reader, spider]) - 1:
                    return text
        except Exception:
            continue
    return None


def get_web_texts(target_urls):
    requests = [(get_web_text, target_url) for target_url in (target_urls if isinstance(target_urls, list) else [target_urls])]
    return {arguments[0]: result for result, name, arguments in manage_thread(requests)}


def get_img_list(target_url, delay=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9"
    }
    for attempt in range(3):
        try:
            print(f"Sending request to {target_url}")
            response = requests.get(target_url, headers=headers, timeout=20)
            soup = BeautifulSoup(response.content, "html.parser")
            img_list = list({
                urljoin(target_url, img.get("src"))
                for img in soup.find_all("img")
                if img.get("src") and urlparse(urljoin(target_url, img.get("src"))).scheme and urlparse(urljoin(target_url, img.get("src"))).netloc and urljoin(target_url, img.get("src")).lower().endswith((".jpg", ".jpeg", ".png"))
            })
            return img_list
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    print("Failed to get a valid response after maximum retries")
    return None


def get_img_lists(target_urls):
    requests = [(get_img_list, target_url) for target_url in (target_urls if isinstance(target_urls, list) else [target_urls])]
    return {arguments[0]: result for result, name, arguments in manage_thread(requests)}
