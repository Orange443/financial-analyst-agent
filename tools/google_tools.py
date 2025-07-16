import os
from dotenv import load_dotenv
import logging
from tavily import TavilyClient

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup Environment ---
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Tool Function ---
def search_google_news(query: str, num_results: int = 5):
    """
    A tool to search for news articles using the Tavily API.

    Args:
        query (str): The search query (e.g., "Reliance Industries stock news").
        num_results (int): The number of search results to return. Defaults to 5.

    Returns:
        list: A list of dictionaries, each containing the title, link, and snippet of a search result.
              Returns an empty list if the API call fails.
    """
    if not TAVILY_API_KEY:
        logging.warning("TAVILY_API_KEY is not configured in .env file. Returning empty list.")
        return []

    try:
        logging.info(f"Performing Tavily search for: '{query}'")
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily.search(query=query, search_depth="advanced", max_results=num_results)
        
        # Extract the relevant parts from the search results.
        formatted_results = [
            {
                "title": item.get('title'),
                "link": item.get('url'),
                "snippet": item.get('content')
            }
            for item in response.get('results', [])
        ]
        
        logging.info(f"Found {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        logging.error(f"An unexpected error occurred during Tavily search: {e}. Returning empty list.")
        return []

# --- Testing Block ---
if __name__ == "__main__":
    logging.info("\n--- Running Test for google_tools.py (with Tavily) ---")
    
    # Example: Searching for news about HDFC Bank
    test_query = "HDFC Bank stock news"
    news_results = search_google_news(query=test_query)
    
    if news_results:
        print(f"\nSuccessfully fetched news for '{test_query}':")
        # Print the title of the first result.
        print(f"Top result: {news_results[0]['title']}")
        print(f"Link: {news_results[0]['link']}")
    else:
        print("\nNo news results returned. This could be due to an error or no results found.")
