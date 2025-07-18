# agents/report_agent.py
import os
import sys
from typing import List, TypedDict, Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.yfinance_tools import get_weekly_stock_data
from tools.google_tools import search_google_news

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class StocksToAnalyze(BaseModel):
    stock_tickers: List[str] = Field(description="A list of 3-5 relevant stock ticker symbols. Must end with '.NS' for Indian stocks.")

class KeyFindings(BaseModel):
    top_performer: Optional[str] = Field(description="The ticker of the best-performing stock.")
    top_performer_change: Optional[float] = Field(description="The percentage change of the top performer.")
    lagging_stock: Optional[str] = Field(description="The ticker of the worst-performing stock.")
    lagging_stock_change: Optional[float] = Field(description="The percentage change of the lagging stock.")
    highest_volume_stock: Optional[str] = Field(description="The ticker of the stock with the highest trading volume.")
    overall_sentiment: str = Field(description="The overall market sentiment (e.g., 'Positive', 'Negative', 'Mixed').")

class FinalReport(BaseModel):
    executive_summary: str = Field(description="A 2-3 sentence executive summary of the sector's performance and outlook.")
    key_findings: KeyFindings = Field(description="The structured key findings.")
    detailed_analysis: str = Field(description="A detailed, multi-paragraph analysis of the sector and each stock in markdown format.")
    expert_outlook: str = Field(description="An expert opinion/outlook section on what these trends might mean for the future.")

class AgentState(TypedDict):
    user_query: str
    stocks: List[str]
    stock_data: Dict[str, List[Dict[str, Any]]]
    news_data: Dict[str, List[Dict[str, Any]]]
    sentiment_data: Dict[str, str]
    report: FinalReport
    error: str

def find_stocks_to_analyze(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Finding Stocks with Gemini 2.5 Flash---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    structured_llm = llm.with_structured_output(schema=StocksToAnalyze)
    prompt = f"You are an expert financial analyst. Based on the query '{state['user_query']}', identify 3-5 relevant Indian stock tickers. Ensure they end with '.NS'."
    result = structured_llm.invoke(prompt)
    return {"stocks": result.stock_tickers}

def get_market_data(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Fetching Market Data---")
    data = {}
    for ticker in state["stocks"]:
        data[ticker] = get_weekly_stock_data(ticker)
    return {"stock_data": data}

def get_news_data(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Fetching News Data---")
    data = {}
    for ticker in state["stocks"]:
        data[ticker] = search_google_news(f"{ticker} stock news")
    return {"news_data": data}

def analyze_sentiment(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Analyzing Sentiment with Gemini 2.5 Flash---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    sentiments = {}
    for ticker, news_items in state["news_data"].items():
        if not news_items:
            sentiments[ticker] = "Neutral"
            continue
        snippets = " ".join([item['snippet'] for item in news_items if item.get('snippet')])
        prompt = f"Based on the following news for {ticker}, classify the sentiment as 'Positive', 'Negative', or 'Neutral'. Answer with only one word.\n\nSnippets: {snippets[:2000]}"
        response = llm.invoke(prompt).content.strip()
        sentiments[ticker] = response
    return {"sentiment_data": sentiments}

def generate_final_report(state: AgentState) -> Dict[str, Any]:
    print("---NODE: Generating Final Report with Gemini 2.5 Pro---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
    structured_llm = llm.with_structured_output(schema=FinalReport)
    
    prompt = f"""
    You are a seasoned financial analyst writing a report. Your tone is insightful, professional, and data-driven.
    
    **Task:** Generate a comprehensive financial report based on the provided data.
    
    **Data:**
    - User Query: {state['user_query']}
    - Stock Price Data: {state['stock_data']}
    - News Sentiment Analysis: {state['sentiment_data']}
    
    **Instructions for Report Generation:**
    1.  **Calculate Key Metrics:**
        - For each stock, calculate the percentage change from the first day's open to the last day's close.
        - Identify the top performer, lagging stock, and the stock with the highest aggregate volume.
    2.  **Executive Summary:** Write a professional, concise summary of the key takeaways and overall market picture.
    3.  **Detailed Analysis:** In markdown format, provide a paragraph analyzing the overall sector performance. Then, for each stock, provide a sub-section with a brief analysis, connecting its price movement to the analyzed news sentiment.
    4.  **Expert Outlook:** Provide a forward-looking opinion. What could these trends indicate? What should an investor watch for? This is your expert take.
    5.  **Crucially, do NOT include any `tool_code` or raw data dumps in your final markdown output.** The analysis should be pure narrative.
    6.  Return the entire analysis in the required JSON format.
    """
    
    report = structured_llm.invoke(prompt)
    return {"report": report}

# --- This is the LangGraph definition the user wants to keep ---
workflow = StateGraph(AgentState)
workflow.add_node("find_stocks", find_stocks_to_analyze)
workflow.add_node("get_market_data", get_market_data)
workflow.add_node("get_news_data", get_news_data)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("generate_report", generate_final_report)

workflow.set_entry_point("find_stocks")
workflow.add_edge("find_stocks", "get_market_data")
workflow.add_edge("get_market_data", "get_news_data")
workflow.add_edge("get_news_data", "analyze_sentiment")
workflow.add_edge("analyze_sentiment", "generate_report")
workflow.add_edge("generate_report", END)

app = workflow.compile()