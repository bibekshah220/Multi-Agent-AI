from lanchain.tools import tool
import requrest
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

