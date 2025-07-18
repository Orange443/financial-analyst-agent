# tools/yfinance_tools.py
import yfinance as yf
import logging
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Enhanced import with error handling
try:
    from dynamic_query_analyzer import EnhancedDynamicQueryAnalyzer
    QUERY_ANALYZER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import EnhancedDynamicQueryAnalyzer: {e}")
    QUERY_ANALYZER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleQueryAnalyzer:
    """Fallback query analyzer if enhanced version fails"""
    
    def __init__(self):
        self.nse_sector_stocks = {
            'fmcg': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO', 'EMAMILTD', 'TATACONSUM'],
            'technology': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTTS'],
            'banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK'],
            'pharmaceutical': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB'],
            'automotive': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO'],
            'energy': ['RELIANCE', 'ONGC', 'IOC', 'BPCL']
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Simple query analysis as fallback"""
        query_lower = query.lower()
        
        # Check for sector keywords
        if 'fmcg' in query_lower or 'consumer' in query_lower:
            return {'type': 'sector', 'sector': 'fmcg', 'confidence': 0.8}
        elif 'it' in query_lower or 'tech' in query_lower:
            return {'type': 'sector', 'sector': 'technology', 'confidence': 0.8}
        elif 'bank' in query_lower:
            return {'type': 'sector', 'sector': 'banking', 'confidence': 0.8}
        elif 'pharma' in query_lower:
            return {'type': 'sector', 'sector': 'pharmaceutical', 'confidence': 0.8}
        elif 'auto' in query_lower:
            return {'type': 'sector', 'sector': 'automotive', 'confidence': 0.8}
        elif 'energy' in query_lower or 'oil' in query_lower:
            return {'type': 'sector', 'sector': 'energy', 'confidence': 0.8}
        
        return {'type': 'general', 'sector': 'fmcg', 'confidence': 0.5}
    
    def discover_tickers(self, query_analysis: Dict) -> List[str]:
        """Discover tickers based on analysis"""
        sector = query_analysis.get('sector', 'fmcg')
        stocks = self.nse_sector_stocks.get(sector, self.nse_sector_stocks['fmcg'])
        return [f"{stock}.NS" for stock in stocks]

class ParallelFinancialDataCollector:
    def __init__(self):
        # Initialize query analyzer with fallback
        if QUERY_ANALYZER_AVAILABLE:
            try:
                self.query_analyzer = EnhancedDynamicQueryAnalyzer()
                logging.info("Enhanced query analyzer initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize enhanced analyzer: {e}")
                self.query_analyzer = SimpleQueryAnalyzer()
                logging.info("Using simple query analyzer as fallback")
        else:
            self.query_analyzer = SimpleQueryAnalyzer()
            logging.info("Using simple query analyzer (enhanced version not available)")
        
        self.max_workers = 10

    def fetch_single_stock_data(self, ticker: str, period='6mo', interval='1d') -> Dict:
        """Fetch data for a single stock"""
        try:
            logging.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            info = stock.info

            if hist.empty:
                logging.warning(f"No data found for ticker: {ticker}")
                return {"status": "error", "message": f"No data for {ticker}", "ticker": ticker}

            # Process historical data
            hist = hist.reset_index()
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            
            hist.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            price_records = hist[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

            # Company info
            company_info = {
                "name": info.get('longName') or info.get('shortName') or ticker,
                "symbol": ticker,
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "marketCap": info.get('marketCap'),
                "regularMarketPrice": info.get('regularMarketPrice') or info.get('currentPrice'),
                "trailingPE": info.get('trailingPE'),
                "forwardPE": info.get('forwardPE'),
                "priceToBook": info.get('priceToBook'),
                "returnOnEquity": info.get('returnOnEquity'),
                "debtToEquity": info.get('debtToEquity'),
                "dividendYield": info.get('dividendYield'),
                "52WeekHigh": info.get('fiftyTwoWeekHigh'),
                "52WeekLow": info.get('fiftyTwoWeekLow'),
                "beta": info.get('beta'),
                "website": info.get('website')
            }

            return {"price_history": price_records, "company_info": company_info, "ticker": ticker}
            
        except Exception as e:
            logging.error(f"Failed for {ticker}: {e}")
            return {"status": "error", "message": str(e), "ticker": ticker}
    
    def fetch_parallel_stock_data(self, tickers: List[str], period='6mo', interval='1d') -> Dict:
        """Fetch data for multiple stocks in parallel"""
        start_time = datetime.now()
        logging.info(f"Starting parallel fetch for {len(tickers)} tickers")
        
        price_data = {}
        fundamentals = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_single_stock_data, ticker, period, interval): ticker 
                for ticker in tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    
                    if result.get('status') == 'error':
                        errors[ticker] = result.get('message', 'Unknown error')
                    else:
                        price_data[ticker] = result['price_history']
                        fundamentals[ticker] = result['company_info']
                        
                except Exception as exc:
                    logging.error(f"Exception for {ticker}: {exc}")
                    errors[ticker] = str(exc)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Parallel fetch completed in {execution_time:.2f} seconds")
        
        return {
            'price_data': price_data,
            'fundamentals': fundamentals,
            'errors': errors,
            'execution_time': execution_time,
            'success_rate': len(fundamentals) / len(tickers) if tickers else 0
        }
    
    def calculate_performance_metrics(self, price_data):
        """Calculate performance metrics"""
        if not price_data or len(price_data) < 2:
            return {
                'daily_change': 0,
                'weekly_change': 0,
                'monthly_change': 0,
                'quarterly_change': 0
            }
        
        current_price = price_data[-1]['close']
        metrics = {}
        
        periods = {
            'daily_change': min(1, len(price_data) - 1),
            'weekly_change': min(5, len(price_data) - 1),
            'monthly_change': min(21, len(price_data) - 1),
            'quarterly_change': min(63, len(price_data) - 1)
        }
        
        for period_name, days_back in periods.items():
            if days_back > 0:
                past_price = price_data[-days_back - 1]['close']
                metrics[period_name] = ((current_price - past_price) / past_price) * 100
            else:
                metrics[period_name] = 0
        
        return metrics
    
    def enrich_fundamentals_data(self, fundamentals, price_data):
        """Enrich fundamental data"""
        enriched = {}
        
        for ticker, info in fundamentals.items():
            enriched[ticker] = info.copy()
            
            if ticker in price_data and price_data[ticker]:
                metrics = self.calculate_performance_metrics(price_data[ticker])
                enriched[ticker].update(metrics)
                
                latest_data = price_data[ticker][-1]
                volume = latest_data.get('volume', 0)
                enriched[ticker]['volume_formatted'] = f"{volume/1000000:.1f}M" if volume >= 1000000 else f"{volume/1000:.1f}K"
            
            # Add basic rating
            weekly_change = enriched[ticker].get('weekly_change', 0)
            if weekly_change > 3:
                enriched[ticker]['rating'] = 'BUY'
            elif weekly_change < -3:
                enriched[ticker]['rating'] = 'SELL'
            else:
                enriched[ticker]['rating'] = 'HOLD'
        
        return enriched
    
    def collect_dynamic_data(self, user_query: str):
        """Main function with error handling"""
        start_time = datetime.now()
        logging.info(f"Processing query: {user_query}")
        
        try:
            # Analyze query
            query_analysis = self.query_analyzer.analyze_query(user_query)
            logging.info(f"Query analysis: {query_analysis}")
            
            # Discover tickers
            tickers = self.query_analyzer.discover_tickers(query_analysis)
            logging.info(f"Discovered {len(tickers)} tickers: {tickers}")
            
            if not tickers:
                return {
                    'error': 'No relevant stocks found for this query',
                    'query_analysis': query_analysis,
                    'tickers': [],
                    'price_data': {},
                    'fundamentals': {},
                    'errors': {}
                }
            
            # Fetch data
            parallel_results = self.fetch_parallel_stock_data(tickers)
            
            # Enrich data
            enriched_fundamentals = self.enrich_fundamentals_data(
                parallel_results['fundamentals'], 
                parallel_results['price_data']
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'query_analysis': query_analysis,
                'tickers': tickers,
                'price_data': parallel_results['price_data'],
                'fundamentals': enriched_fundamentals,
                'errors': parallel_results['errors'],
                'performance_metrics': {
                    'total_execution_time': total_time,
                    'parallel_fetch_time': parallel_results['execution_time'],
                    'success_rate': parallel_results['success_rate'],
                    'companies_processed': len(tickers),
                    'successful_fetches': len(enriched_fundamentals),
                    'failed_fetches': len(parallel_results['errors'])
                },
                'data_quality': {
                    'total_companies': len(tickers),
                    'successful_data_fetch': len(enriched_fundamentals),
                    'data_completeness': len(enriched_fundamentals) / len(tickers) if tickers else 0,
                    'overall_quality': 'High' if len(enriched_fundamentals) > 0 else 'Low'
                },
                'collection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error in collect_dynamic_data: {e}")
            return {
                'error': f'Data collection failed: {str(e)}',
                'query_analysis': {'type': 'general', 'sector': 'fmcg', 'confidence': 0.5},
                'tickers': [],
                'price_data': {},
                'fundamentals': {},
                'errors': {'system': str(e)}
            }

# Initialize global collector
parallel_financial_collector = ParallelFinancialDataCollector()

# Maintain backward compatibility
def get_stock_analysis(query):
    """Main function for stock analysis"""
    return parallel_financial_collector.collect_dynamic_data(query)

def get_sector_data(tickers, period='6mo', interval='1d'):
    """Legacy function with parallel processing"""
    collector = ParallelFinancialDataCollector()
    parallel_results = collector.fetch_parallel_stock_data(tickers, period, interval)
    
    enriched_fundamentals = collector.enrich_fundamentals_data(
        parallel_results['fundamentals'], 
        parallel_results['price_data']
    )
    
    return {
        'price_data': parallel_results['price_data'],
        'fundamentals': enriched_fundamentals,
        'errors': parallel_results['errors']
    }

def get_weekly_stock_data(ticker: str):
    """Legacy function for compatibility"""
    try:
        result = parallel_financial_collector.fetch_single_stock_data(ticker, period='7d', interval='1d')
        if 'price_history' in result:
            return result['price_history']
        else:
            return {"status": "error", "message": result.get('message', 'Unknown error')}
    except Exception as e:
        return {"status": "error", "message": str(e)}
