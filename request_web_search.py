import requests
import random
from scraper import purify
from aife_time import now, days_ago, year_start, year_end, iso_date
from aife_utils import retrieve

JINA_API_KEY = retrieve("Jina")

TAVILY_API_KEYS = [
    retrieve("Tavily"),
    retrieve("Tavily2"),
    retrieve("Tavily3"),
    retrieve("Tavily4")
]

BING_API_KEYS = [
    retrieve("BingKey"),
    retrieve("Bing2Key")
]

SERP_API_KEYS = [
    retrieve("Serp"),
    retrieve("Serp2"),
    retrieve("Serp3"),
    retrieve("Serp4")
]

SERPER_API_KEY = retrieve("Serper")

EXA_API_KEY = retrieve("Exa")


def reader_search(query):
    url = f"https://s.jina.ai/{query}"
    headers = {
        "Authorization": JINA_API_KEY
    }
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, headers=headers, timeout=20)
            if response.text:
                return purify(response.text)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return ""


def request_tavily_answer(url, payload):
    for attempt, api_key in enumerate(random.sample(TAVILY_API_KEYS, len(TAVILY_API_KEYS)), 1):
        payload["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, timeout=20).json()
            results = response.get("answer")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return ""


def tavily_answer(query):
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": "general",
        "include_answer": True
    }
    return request_tavily_answer(url, payload)


def request_tavily(url, payload):
    for attempt, api_key in enumerate(random.sample(TAVILY_API_KEYS, len(TAVILY_API_KEYS)), 1):
        payload["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, json=payload, timeout=20).json()
            results = response.get("results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def tavily_news_by_freshness(query, freshness=None):
    url = "https://api.tavily.com/search"
    payload = {
        "query": query,
        "search_depth": "advanced",
        "topic": "news",
        "days": freshness,
        "include_answer": False,
        "max_results": 50
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("content"),
        "url": item.get("url"),
        "date": iso_date(item.get("published_date"))
    } for index, item in enumerate(request_tavily(url, payload), start=1)}


def request_bing(url, params):
    for attempt, api_key in enumerate(random.sample(BING_API_KEYS, len(BING_API_KEYS)), 1):
        headers = {
            "Ocp-Apim-Subscription-Key": api_key
        }
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, headers=headers, params=params, timeout=10).json()
            results = response.get("webPages", {}).get("value")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def bing_by_freshness(query, freshness=None):
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "responseFilter": "Webpages",
        "freshness": {1: "Day", 7: "Week", 30: "Month"}.get(freshness),
        "mkt": "en-US",
        "safeSearch": "Off",
        "count": 50
    }
    return {index: {
        "title": item.get("name"),
        "summary": item.get("snippet"),
        "url": item.get("url"),
        "date": iso_date(item.get("datePublished"))
    } for index, item in enumerate(request_bing(url, params), start=1)}


def bing_by_year_range(query, year_range=None):
    url = "https://api.bing.microsoft.com/v7.0/search"
    params = {
        "q": query,
        "responseFilter": "Webpages",
        "freshness": f"{year_start(year_range[0]).date().isoformat()}..{year_end(year_range[1]).date().isoformat()}" if year_range else None,
        "mkt": "en-US",
        "safeSearch": "Off",
        "count": 50
    }
    return {index: {
        "title": item.get("name"),
        "summary": item.get("snippet"),
        "url": item.get("url"),
        "date": iso_date(item.get("datePublished"))
    } for index, item in enumerate(request_bing(url, params), start=1)}


def request_serp(url, params):
    for attempt, api_key in enumerate(random.sample(SERP_API_KEYS, len(SERP_API_KEYS)), 1):
        params["api_key"] = api_key
        try:
            print(f"Sending request to {url}")
            response = requests.get(url, params=params, timeout=10).json()
            results = response.get("organic_results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def duckduckgo_by_freshness(query, freshness=None):
    url = "https://serpapi.com/search.json?engine=duckduckgo"
    params = {
        "q": query,
        "df": {1: "d", 7: "w", 30: "m"}.get(freshness),
        "safe": -2,
        "no_cache": "true",
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for index, item in enumerate(request_serp(url, params), start=1)}


def duckduckgo_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=duckduckgo"
    params = {
        "q": query,
        "df": f"{year_start(year_range[0]).date().isoformat()}..{year_end(year_range[1]).date().isoformat()}" if year_range else None,
        "safe": -2,
        "no_cache": "true",
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for index, item in enumerate(request_serp(url, params), start=1)}


def google_scholar_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=google_scholar"
    params = {
        "q": query,
        "as_ylo": year_range[0] if year_range else None,
        "as_yhi": year_range[1] if year_range else None,
        "location": "County Dublin,Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true",
        "num": 20
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "publication_info": item.get("publication_info", {}).get("summary"),
        "url": item.get("link")
    } for index, item in enumerate(request_serp(url, params), start=1)}


def google_patents_by_year_range(query, year_range=None):
    url = "https://serpapi.com/search.json?engine=google_patents"
    params = {
        "q": query,
        "after": f"publication:{year_start(year_range[0]).year}{year_start(year_range[0]).month:02d}{year_start(year_range[0]).day:02d}" if year_range else None,
        "before": f"publication:{year_end(year_range[1]).year}{year_end(year_range[1]).month:02d}{year_end(year_range[1]).day:02d}" if year_range else None,
        "status": "GRANT",
        "type": "PATENT",
        "no_cache": "true",
        "num": 20
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("pdf"),
        "grant_date": iso_date(item.get("grant_date"))
    } for index, item in enumerate(request_serp(url, params), start=1)}


def request_serper(url, headers, payload):
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=10).json()
            results = response.get("organic")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def google_by_freshness(query, freshness=None):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY
    }
    payload = {
        "q": query,
        "tbs": {1: "qdr:d", 7: "qdr:w", 30: "qdr:m"}.get(freshness),
        "location": "Dublin City, County Dublin, Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true",
        "num": 100
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("snippet"),
        "url": item.get("link"),
        "date": iso_date(item.get("date"))
    } for index, item in enumerate(request_serper(url, headers, payload), start=1)}


def request_serper_answer(url, headers, payload):
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=10).json()
            answer_box = response.get("answerBox")
            knowledge_graph = response.get("knowledgeGraph")
            results = response.get("organic")
            if answer_box or knowledge_graph or results:
                return {"answerBox": answer_box, "knowledgeGraph": knowledge_graph, "results": results}
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return {}


def google_answer(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY
    }
    payload = {
        "q": query,
        "location": "Dublin City, County Dublin, Ireland",
        "gl": "ie",
        "safe": "off",
        "no_cache": "true"
    }
    return request_serper_answer(url, headers, payload)


def request_exa(url, headers, payload):
    for attempt in range(3):
        try:
            print(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=payload, timeout=20).json()
            results = response.get("results")
            if results:
                return results
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
    print("Failed to get a valid response after maximum retries")
    return []


def exa_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("summary"),
        "author": item.get("author"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for index, item in enumerate(request_exa(url, headers, payload), start=1)}


def exa_by_year_range(query, year_range=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{year_start(year_range[0]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "endPublishedDate": f"{year_end(year_range[1]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "numResults": 25
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("summary"),
        "author": item.get("author"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for index, item in enumerate(request_exa(url, headers, payload), start=1)}


def exa_news_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "news",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("summary"),
        "author": item.get("author"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for index, item in enumerate(request_exa(url, headers, payload), start=1)}


def exa_paper_by_freshness(query, freshness=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "research paper",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{days_ago(freshness).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if freshness else None,
        "endPublishedDate": f"{now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z",
        "numResults": 25
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("summary"),
        "author": item.get("author"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for index, item in enumerate(request_exa(url, headers, payload), start=1)}


def exa_paper_by_year_range(query, year_range=None):
    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": EXA_API_KEY
    }
    payload = {
        "query": query,
        "category": "research paper",
        "contents": {
            "summary": {"query": query}
        },
        "useAutoprompt": True,
        "type": "auto",
        "startPublishedDate": f"{year_start(year_range[0]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "endPublishedDate": f"{year_end(year_range[1]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z" if year_range else None,
        "numResults": 25
    }
    return {index: {
        "title": item.get("title"),
        "summary": item.get("summary"),
        "author": item.get("author"),
        "url": item.get("url"),
        "date": iso_date(item.get("publishedDate"))
    } for index, item in enumerate(request_exa(url, headers, payload), start=1)}
