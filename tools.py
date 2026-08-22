from lanchain.tools import tool
import requrest
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
from rich import print

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a given topic.
    Return the title, URL, and snippet for each relevant result.
    """
    try:
        results = tavily.search(query-query,max_results=5)

        if not results:
            return "No results found."

        return "\n\n".join(
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL: {result.get('url', 'N/A')}\n"
            f"Snippet: {result.get('content', 'N/A')}"
            for result in results
        )

    except Exception as e:
        return f"An error occurred during the web search: {str(e)}"
