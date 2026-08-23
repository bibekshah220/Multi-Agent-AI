from langchain.tools import tool
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic.
    Returns the title, URL, and a short snippet for each result.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"An error occurred during the web search: {str(e)}"

    if not results:
        return "No results found."

    return "\n\n".join(
        f"Title: {r.get('title', 'N/A')}\n"
        f"URL: {r.get('href', 'N/A')}\n"
        f"Snippet: {r.get('body', 'N/A')[:300]}"
        for r in results
    )


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


