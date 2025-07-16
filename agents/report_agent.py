import os
import sys
from typing import List, TypedDict, Dict, Any

# --- Path Correction ---
# This is a best practice to ensure that the script can find the 'tools' module
# even when run from the 'agents' directory. It adds the parent directory 
# (the main project folder) to Python's system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# --- Pydantic Import Fix ---
# As per the deprecation warning, we now import directly from pydantic.
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Import our custom tools
from tools.yfinance_tools import get_weekly_stock_data
from tools.google_tools import search_google_news

# --- Load Environment Variables ---
load_dotenv()
# We will use a single, unified key for all Google services.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of our financial analyst agent.
    """
    user_query: str
    stocks: List[str]
    stock_data: Dict[str, List[Dict[str, Any]]]
    news_data: Dict[str, List[Dict[str, Any]]]
    analysis: str
    error: str

# --- Pydantic Models for Structured Output ---
class StocksToAnalyze(BaseModel):
    """
    A Pydantic model that defines the structure for the stocks we want to analyze.
    """
    stock_tickers: List[str] = Field(
        description="A list of 3-5 relevant stock ticker symbols based on the user's query. For Indian stocks, ensure they end with '.NS'."
    )

# --- Agent Nodes ---

def find_stocks_to_analyze(state: AgentState) -> Dict[str, Any]:
    """
    First node: Takes the user's query and uses an LLM to identify a list of
    relevant stock tickers to analyze.
    """
    print("---Finding Stocks to Analyze---")
    user_query = state["user_query"]
    # **THE FIX IS HERE**: Updated to a newer, recommended model name.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Use the .with_structured_output() method to force the LLM to return
    # data in the format of our StocksToAnalyze Pydantic model.
    structured_llm = llm.with_structured_output(schema=StocksToAnalyze)
    
    prompt = f"""
    You are an expert financial analyst. A user has asked the following query: "{user_query}".
    
    Based on this query, please identify a list of 3 to 5 of the most relevant and important
    publicly traded Indian companies. Provide their stock ticker symbols.
    
    Important: For all Indian stocks, the ticker symbol must end with the '.NS' suffix.
    For example, Reliance Industries is 'RELIANCE.NS'.
    """
    result = structured_llm.invoke(prompt)
    stock_tickers = result.stock_tickers
    print(f"Stocks found: {stock_tickers}")
    return {"stocks": stock_tickers}

def get_market_data(state: AgentState) -> Dict[str, Any]:
    """
    Second node: Fetches the weekly stock data for each stock identified in the previous step.
    """
    print("---Fetching Market Data---")
    stocks = state["stocks"]
    data = {}
    for ticker in stocks:
        # Call our yfinance tool for each ticker
        stock_data = get_weekly_stock_data(ticker)
        data[ticker] = stock_data
    return {"stock_data": data}

def get_news_data(state: AgentState) -> Dict[str, Any]:
    """
    Third node: Fetches recent news articles for each stock.
    """
    print("---Fetching News Data---")
    stocks = state["stocks"]
    data = {}
    for ticker in stocks:
        # Format a search query for the Google tool
        query = f"{ticker} stock news"
        news_data = search_google_news(query)
        data[ticker] = news_data
    return {"news_data": data}

def analyze_and_summarize(state: AgentState) -> Dict[str, Any]:
    """
    Final node: Takes all the collected data (prices and news) and uses an LLM
    to generate a final, comprehensive analysis.
    """
    print("---Analyzing and Summarizing Data---")
    stock_data = state["stock_data"]
    news_data = state["news_data"]

    # --- Prompt Generation with Disclaimer for Missing News ---
    
    # Start building the prompt with the initial data.
    
    prompt_data = ""
    
    news_unavailable = False
    
    for ticker, data in stock_data.items():
        
        prompt_data += f"\n--- Stock: {ticker} ---\n"
        
        prompt_data += f"Weekly Price Data: {data}\n"
        
        # Check if news data is available and not empty.
        
        if news_data.get(ticker):
            
            prompt_data += f"Recent News: {news_data[ticker]}\n"
        
        else:
            
            # If news is not available for any stock, set the flag.
            
            news_unavailable = True

    
    # --- LLM and Prompt Configuration ---
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Base prompt for the analysis.
    
    prompt = f"""
    You are an expert financial analyst. Your task is to generate a comprehensive, 
    easy-to-read summary and analysis based on the following weekly stock data and news.

    Your analysis should include:
    1.  An overall summary of the sector's performance.
    2.  A highlight of the best and worst-performing stock from the list.
    3.  A brief, neutral summary for each individual stock, mentioning any key news that
        might have influenced its performance.
    """

    # --- Add Disclaimer if News Data is Missing ---
    
    if news_unavailable:
        
        prompt += "\n**Important Note:** News data could not be retrieved for some or all stocks. Please state this in your analysis and proceed with the analysis based on the available price data. Acknowledge that the lack of news context limits the depth of the analysis."

    # Add the collected data to the prompt.
    
    prompt += f"\n\nPlease analyze the following data:\n{prompt_data}"

    # --- Invoke LLM and Return Analysis ---
    
    analysis = llm.invoke(prompt).content
    
    print("---Analysis Complete---")
    
    return {"analysis": analysis}


# --- Graph Definition ---

# This is where we define the workflow of our agent.
workflow = StateGraph(AgentState)

# Add the nodes (the functions we defined above) to the graph.
workflow.add_node("find_stocks", find_stocks_to_analyze)
workflow.add_node("get_market_data", get_market_data)
workflow.add_node("get_news_data", get_news_data)
workflow.add_node("analyze_data", analyze_and_summarize)

# Define the edges (the connections between the nodes).
# This tells the agent the order in which to perform the steps.
workflow.set_entry_point("find_stocks")
workflow.add_edge("find_stocks", "get_market_data")
workflow.add_edge("get_market_data", "get_news_data")
workflow.add_edge("get_news_data", "analyze_data")
workflow.add_edge("analyze_data", END) # The END state signifies the workflow is complete.

# Compile the graph into a runnable application.
app = workflow.compile()


# --- Testing Block ---
# This allows us to run the agent directly from the terminal for testing.
if __name__ == "__main__":
    print("--- Starting Financial Analyst Agent ---")
    
    # Define the initial state for the agent run.
    initial_state = {
        "user_query": "Analyze the top Indian IT sector stocks for the last week",
        # Initialize other fields as empty
        "stocks": [],
        "stock_data": {},
        "news_data": {},
        "analysis": "",
        "error": None
    }
    
    # Run the agent with the initial state.
    final_state = app.invoke(initial_state)
    
    # Print the final analysis generated by the agent.
    print("\n\n--- FINAL REPORT ---")
    print(final_state['analysis'])

