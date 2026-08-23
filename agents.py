from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)

SYSTEM_PROMPT = (
    "You are a helpful research assistant. "
    "Use web_search to find recent information and scrape_url to read a page "
    "in depth. Always cite the URLs you used."
)

agent = create_agent(
    model=llm,
    tools=[web_search, scrape_url],
    system_prompt=SYSTEM_PROMPT,
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What are the latest developments in multi-agent AI systems?"}]}
    )
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    print(content)