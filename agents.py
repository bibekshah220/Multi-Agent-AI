from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0)

# Shared retry config: back off on transient 429 rate-limit errors.
RETRY_KWARGS = dict(
    retry_if_exception_type=(GoogleRateLimitError,),
    wait_exponential_jitter=True,
    exponential_jitter_params={"initial": 2, "max": 60, "jitter": 2},
    stop_after_attempt=5,
)


#1st step
RESEARCH_SYSTEM_PROMPT = (
    "You are a research agent. Use web_search AT MOST twice to find sources, then "
    "optionally use scrape_url on AT MOST 2 of the most promising URLs. "
    "Do not keep searching or scraping in a loop. As soon as you have enough "
    "information, STOP calling tools and return a concise summary of the key facts "
    "with their source URLs."
)

READER_SYSTEM_PROMPT = (
    "You are a reader agent. Use scrape_url on AT MOST 3 of the most relevant URLs "
    "from the provided research. Do not loop. Once you have read them, STOP calling "
    "tools and return the additional detail and context you extracted."
)


def build_agent():
    return create_agent(
        model=llm,
        tools=[web_search, scrape_url],
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    ).with_retry(**RETRY_KWARGS)

# 2nd agent

def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    ).with_retry(**RETRY_KWARGS)

# writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = (writer_prompt | llm | StrOutputParser()).with_retry(**RETRY_KWARGS)

#critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical research analyst. Evaluate the quality and reliability of the research provided."),  
    ("human", """Critically evaluate the research below.

Topic: {topic}

Research Gathered:
{research}

Assess the research on:
- Accuracy and factual correctness
- Completeness (are important aspects missing?)
- Source reliability (are the URLs credible?)
- Bias or unsupported claims

Then provide:
- A numeric score from 1 to 10 (formatted as "Score: X/10")
- A short overall verdict (Strong / Adequate / Weak)
- Specific gaps or issues found
- Concrete suggestions to improve the research."""),
])

critic_chain = (critic_prompt | llm | StrOutputParser()).with_retry(**RETRY_KWARGS)
