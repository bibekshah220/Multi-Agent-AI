from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic.
    Return title, URL, and a short snippet for each result.
    """
    results = tavily.search(
        query=query,
        max_results=5
    )

    out = []

    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['content'][:300]}"
        )

    return "\n\n".join(out)
