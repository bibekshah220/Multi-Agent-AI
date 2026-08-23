from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)


#1st step
def build_agent():
    return create_agent(
        model=llm,
        tools=[web_search, scrape_url],
    )

# 2nd agent

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
    )





