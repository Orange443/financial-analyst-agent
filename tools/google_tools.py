# tools/google_tools.py
import os
from dotenv import load_dotenv
import logging
from tavily import TavilyClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class DynamicNewsCollector:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    
    def collect_news_for_tickers(self, tickers, fundamentals_data):
        """Collect news for multiple tickers dynamically"""
        news_data = {}
        
        if not self.tavily_client:
            logging.warning("Tavily API key not configured")
            return news_data
        
        for ticker in tickers:
            # Get company name for better search
            company_name = self.get_company_name(ticker, fundamentals_data)
            
            # Search for news
            news_results = self.search_company_news(company_name, ticker)
            if news_results:
                news_data[ticker] = news_results
        
        return news_data
    
    def get_company_name(self, ticker, fundamentals_data):
        """Extract company name from fundamentals data"""
        if ticker in fundamentals_data:
            return fundamentals_data[ticker].get('name', ticker.replace('.NS', ''))
        return ticker.replace('.NS', '')
    
    def search_company_news(self, company_name, ticker, num_results=3):
        """Search for news about a specific company"""
        try:
            # Create search query
            query = f"{company_name} stock news earnings financial results"
            
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results
            )
            
            # Format results
            formatted_results = []
            for item in response.get('results', []):
                formatted_results.append({
                    "title": item.get('title'),
                    "link": item.get('url'),
                    "snippet": item.get('content') or item.get('description', 'No preview available'),
                    "relevance_score": item.get('score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error searching news for {company_name}: {e}")
            return []
    
    def analyze_news_sentiment(self, news_articles):
        """Analyze sentiment of news articles"""
        if not news_articles:
            return 'Neutral'
        
        positive_words = ['growth', 'profit', 'gain', 'rise', 'strong', 'positive', 'bullish', 'upgrade']
        negative_words = ['loss', 'fall', 'decline', 'weak', 'negative', 'bearish', 'downgrade', 'concern']
        
        sentiment_score = 0
        for article in news_articles:
            text = (article.get('title', '') + ' ' + article.get('snippet', '')).lower()
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            sentiment_score += positive_count - negative_count
        
        if sentiment_score > 0:
            return 'Positive'
        elif sentiment_score < 0:
            return 'Negative'
        else:
            return 'Neutral'

# Backward compatibility
def search_google_news(query, num_results=5):
    """Legacy function for backward compatibility"""
    collector = DynamicNewsCollector()
    if not collector.tavily_client:
        return []
    
    try:
        response = collector.tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=num_results
        )
        
        return [
            {
                "title": item.get('title'),
                "link": item.get('url'),
                "snippet": item.get('content') or item.get('description', 'No preview available')
            }
            for item in response.get('results', [])
        ]
    except Exception as e:
        logging.error(f"Error in legacy news search: {e}")
        return []
