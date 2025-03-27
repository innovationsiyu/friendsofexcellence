import subprocess
import sys
from markdown_it import MarkdownIt
from urllib.parse import urljoin
from validators import url
import requests
import random
from io import BytesIO
from PIL import Image
import hashlib
from playwright.sync_api import sync_playwright
from time import sleep
import time
import os
import asyncio
from playwright.async_api import async_playwright
from aife_utils import retrieve, RE_NORMALIZE_NEWLINES, RE_REMOVE_MARKDOWN_COMPOSITE_LINKS, RE_REMOVE_MARKDOWN_BASIC_LINKS, RE_REMOVE_HTML_TAGS, RE_REMOVE_INVALID_LINES, RE_COMPRESS_NEWLINES, manage_futures, manage_futures_async, now_and_choices


# subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)

FIRECRAWL_API_KEYS = [
    retrieve("Firecrawl7"),
    retrieve("Firecrawl8"),
    retrieve("Firecrawl9")
]

SPIDER_API_KEY = retrieve("Spider")


def purify(text):
    text = RE_NORMALIZE_NEWLINES.sub("\n", text)
    text = RE_REMOVE_MARKDOWN_COMPOSITE_LINKS.sub("", text)
    text = RE_REMOVE_MARKDOWN_BASIC_LINKS.sub("", text)
    text = RE_REMOVE_HTML_TAGS.sub("", text)
    text = RE_REMOVE_INVALID_LINES.sub("", text)
    text = RE_COMPRESS_NEWLINES.sub("\n", text)
    return text[:50000].strip()


def tidy(text):
    text = RE_NORMALIZE_NEWLINES.sub("\n", text)
    text = RE_REMOVE_INVALID_LINES.sub("", text)
    text = RE_COMPRESS_NEWLINES.sub("\n", text)
    return text[:50000].strip()


def get_lines_and_image_urls(web_url, web_content):
    lines = list(dict.fromkeys(line for line in (line.strip() for line in web_content.splitlines()) if line))
    md = MarkdownIt()
    i = 0
    while i < len(lines):
        items = []
        for child in [child for token in md.parse(lines[i]) if token.type == "inline" for child in token.children]:
            if child.type == "image":
                try:
                    items.append(urljoin(web_url, child.attrs.get("src").lstrip()))
                except Exception:
                    continue
            elif child.type == "text":
                if content := child.content.strip():
                    items.append(content)
        if items:
            lines[i:i + 1] = items
            i += len(items)
        else:
            i += 1
    return lines


def get_lines(body_content):
    return list(dict.fromkeys(line for line in (line.strip() for line in body_content.splitlines()) if line))


def get_image(url):
    try:
        with Image.open(BytesIO(requests.get(url, timeout=(30, 120)).content)) as f:
            if max(f.size) < 100:
                return None
            if min(f.size) > 1024:
                ratio = 1024 / min(f.size)
                f = f.resize((int(f.size[0] * ratio), int(f.size[1] * ratio)), Image.Resampling.LANCZOS)
            image_format = f.format if f.format in ["JPEG", "PNG"] else "JPEG"
            if image_format == "JPEG" and f.mode != "RGB":
                f = f.convert("RGB")
            image_path = f"temp-images/{now_and_choices()}.{image_format.lower()}"
            buffer = BytesIO()
            f.save(buffer, format=image_format)
            image_binary = buffer.getvalue()
            with open(image_path, "wb") as out_f:
                out_f.write(image_binary)
            return image_path
    except Exception as e:
        print(f"Error in get_image from {url}: {e}")
        return None


def insert_image_paths(lines):
    if url_indices := [i for i, line in enumerate(lines) if url(line)]:
        requests = [(get_image, (lines[i],)) for i in url_indices]
        image_url_and_path_dict = {arguments[0]: result for result, function, arguments in manage_futures(requests)}
        for i in url_indices:
            if image_url_and_path_dict[lines[i]]:
                lines[i] = image_url_and_path_dict[lines[i]]
    return lines


def tidy_body_content(body_content):
    return insert_image_paths(get_lines(tidy(body_content)))


def tidy_body_contents(body_contents):
    requests = [(tidy_body_content, (body_content,)) for body_content in (body_contents if isinstance(body_contents, list) else [body_contents])]
    return {arguments[0]: result for result, function, arguments in manage_futures(requests)}


def firecrawl(web_url):
    url = "https://api.firecrawl.dev/v1/scrape"
    payload = {
        "url": web_url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "actions": [
            {
                "type": "wait",
                "milliseconds": 5000
            },
            {
                "type": "scroll",
                "direction": "down"
            }
        ]
    }
    for attempt, api_key in enumerate(random.sample(FIRECRAWL_API_KEYS, 3)):
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, headers=headers, timeout=30).json()
            if content := response.get("data", {}).get("markdown"):
                return content
        except Exception as e:
            print(f"Firecrawl attempt {attempt + 1} failed: {e}")
    print(f"Firecrawl failed to get a valid response after maximum retries")
    return None


def spider(web_url):
    url = "https://api.spider.cloud/crawl"
    headers = {
        "Authorization": f"Bearer {SPIDER_API_KEY}",
    }
    json_data = {
        "url": web_url,
        "limit": 1,
        "return_format": "markdown"
    }
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=json_data, timeout=30).json()
            if content := response[0].get("content"):
                return content
        except Exception as e:
            print(f"Spider attempt {attempt + 1} failed: {e}")
    print(f"Spider failed to get a valid response after maximum retries")
    return None


def reader(web_url):
    url = f"https://r.jina.ai/{web_url}"
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, timeout=30)
            if text := response.text:
                return text
        except Exception as e:
            print(f"Reader attempt {attempt + 1} failed: {e}")
    print("Reader failed to get a valid response after maximum retries")
    return None


def get_web_content(web_url):
    if web_url:
        for i, function in enumerate([firecrawl, spider]):
            if web_content := function(web_url):
                if len(web_content) >= 600 or i == len([firecrawl, spider]) - 1:
                    return insert_image_paths(get_lines_and_image_urls(web_url, tidy(web_content)))
    return None


def get_web_contents(web_urls):
    requests = [(get_web_content, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in manage_futures(requests)}


def get_web_text(web_url):
    if web_url:
        for i, function in enumerate([reader, spider]):
            if web_text := function(web_url):
                if len(web_text) >= 600 or i == len([reader, spider]) - 1:
                    return purify(web_text)
    return None


def get_web_texts(web_urls):
    requests = [(get_web_text, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in manage_futures(requests)}

"""
def get_web_screenshot(web_url):
    try:
        print(f"Taking screenshot of {web_url}")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox"]
            )
            page = browser.new_page(
                viewport={"width": 1280, "height": 1000},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                extra_http_headers={"Accept-Language": "en,zh-CN;q=0.9,zh;q=0.8"}
            )
            page.set_default_navigation_timeout(120000)
            for attempt in range(3):
                try:
                    page.goto(web_url, wait_until="domcontentloaded", timeout=60000)
                    break
                except Exception as e:
                    print(f"Loading page attempt {attempt + 1} failed: {e}")
            else:
                print("Failed to load the page after maximum retries")
                return None
            height = 0
            for attempt in range(50):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                sleep(1)
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == height:
                    if page.evaluate("Array.from(document.getElementsByTagName('img')).every(img => img.complete && (img.naturalWidth > 0 || img.naturalHeight > 0))"):
                        break
                height = new_height
            image_path = os.path.join("temp-images", f"{now_and_choices()}.png")
            page.screenshot(path=image_path, full_page=True)
            page.close()
            browser.close()
            return image_path
    except Exception as e:
        print(f"An error occurred during the screenshot: {e}")
        return None


def get_web_screenshots(web_urls):
    requests = [(get_web_screenshot, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in manage_futures(requests, 3)}


async def get_web_screenshot(web_url):
    try:
        print(f"Taking screenshot of {web_url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--no-sandbox"]
            )
            page = await browser.new_page(
                viewport={"width": 1280, "height": 1000},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                extra_http_headers={"Accept-Language": "en,zh-CN;q=0.9,zh;q=0.8"}
            )
            page.set_default_navigation_timeout(120000)
            for attempt in range(3):
                try:
                    await page.goto(web_url, wait_until="domcontentloaded", timeout=60000)
                    break
                except Exception as e:
                    print(f"Loading page attempt {attempt + 1} failed: {e}")
            else:
                print("Failed to load the page after maximum retries")
                return None
            height = 0
            for attempt in range(50):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1)
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height == height:
                    if await page.evaluate("Array.from(document.getElementsByTagName('img')).every(img => img.complete && (img.naturalWidth > 0 || img.naturalHeight > 0))"):
                        break
                height = new_height
            image_path = os.path.join("temp-images", f"{now_and_choices()}.png")
            await page.screenshot(path=image_path, full_page=True)
            await page.close()
            await browser.close()
            return image_path
    except Exception as e:
        print(f"An error occurred during the screenshot: {e}")
        return None


async def get_web_screenshots(web_urls):
    requests = [(get_web_screenshot, (web_url,)) for web_url in (web_urls if isinstance(web_urls, list) else [web_urls])]
    return {arguments[0]: result for result, function, arguments in await manage_futures_async(requests, 10)}
"""