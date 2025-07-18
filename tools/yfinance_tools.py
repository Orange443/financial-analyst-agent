# tools/yfinance_tools.py
import yfinance as yf
import logging
import concurrent.futures
from datetime import datetime, timedelta
from dynamic_query_analyzer import EnhancedDynamicQueryAnalyzer
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParallelFinancialDataCollector:
    def __init__(self):
        self.query_analyzer = EnhancedDynamicQueryAnalyzer()
        self.max_workers = 10  # Configurable thread pool size
    
    def fetch_single_stock_data(self, ticker: str, period='6mo', interval='1d') -> Dict:
        """Fetch data for a single stock (optimized for parallel execution)"""
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

            # Enhanced company info
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
                "profitMargins": info.get('profitMargins'),
                "operatingMargins": info.get('operatingMargins'),
                "earningsGrowth": info.get('earningsGrowth'),
                "revenueGrowth": info.get('revenueGrowth'),
                "website": info.get('website')
            }

            logging.info(f"Successfully fetched data for {ticker}")
            return {"price_history": price_records, "company_info": company_info, "ticker": ticker}
            
        except Exception as e:
            logging.error(f"Failed for {ticker}: {e}")
            return {"status": "error", "message": str(e), "ticker": ticker}
    
    def fetch_parallel_stock_data(self, tickers: List[str], period='6mo', interval='1d') -> Dict:
        """Fetch data for multiple stocks in parallel"""
        start_time = datetime.now()
        logging.info(f"Starting parallel fetch for {len(tickers)} tickers with {self.max_workers} workers")
        
        price_data = {}
        fundamentals = {}
        errors = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.fetch_single_stock_data, ticker, period, interval): ticker 
                for ticker in tickers
            }
            
            # Collect results as they complete
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
        
        # Performance metrics
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        logging.info(f"Parallel fetch completed in {execution_time:.2f} seconds")
        logging.info(f"Success: {len(fundamentals)}, Errors: {len(errors)}")
        
        return {
            'price_data': price_data,
            'fundamentals': fundamentals,
            'errors': errors,
            'execution_time': execution_time,
            'success_rate': len(fundamentals) / len(tickers) if tickers else 0
        }
    
    def calculate_performance_metrics(self, price_data):
        """Calculate comprehensive performance metrics"""
        if not price_data or len(price_data) < 2:
            return {
                'daily_change': 0,
                'weekly_change': 0,
                'monthly_change': 0,
                'quarterly_change': 0,
                'ytd_change': 0,
                'volatility': 0
            }
        
        current_price = price_data[-1]['close']
        metrics = {}
        
        # Calculate different period returns
        periods = {
            'daily_change': min(1, len(price_data) - 1),
            'weekly_change': min(5, len(price_data) - 1),
            'monthly_change': min(21, len(price_data) - 1),
            'quarterly_change': min(63, len(price_data) - 1),
            'ytd_change': min(252, len(price_data) - 1)
        }
        
        for period_name, days_back in periods.items():
            if days_back > 0:
                past_price = price_data[-days_back - 1]['close']
                metrics[period_name] = ((current_price - past_price) / past_price) * 100
            else:
                metrics[period_name] = 0
        
        # Calculate volatility (standard deviation of returns)
        if len(price_data) >= 20:
            closes = [p['close'] for p in price_data[-20:]]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            metrics['volatility'] = (variance ** 0.5) * 100
        else:
            metrics['volatility'] = 0
        
        return metrics
    
    def enrich_fundamentals_data(self, fundamentals, price_data):
        """Enrich fundamental data with calculated metrics"""
        enriched = {}
        
        for ticker, info in fundamentals.items():
            enriched[ticker] = info.copy()
            
            # Add performance metrics
            if ticker in price_data and price_data[ticker]:
                metrics = self.calculate_performance_metrics(price_data[ticker])
                enriched[ticker].update(metrics)
                
                # Format volume
                latest_data = price_data[ticker][-1]
                volume = latest_data.get('volume', 0)
                enriched[ticker]['volume_formatted'] = self.format_volume(volume)
                
                # Add technical indicators
                enriched[ticker].update(self.calculate_technical_indicators(price_data[ticker]))
            
            # Add enhanced rating
            enriched[ticker]['rating'] = self.determine_enhanced_rating(enriched[ticker])
            
            # Add news sentiment placeholder
            enriched[ticker]['news_sentiment'] = 'Neutral'
        
        return enriched
    
    def format_volume(self, volume):
        """Format volume in readable format"""
        if volume >= 1000000:
            return f"{volume/1000000:.1f}M"
        elif volume >= 1000:
            return f"{volume/1000:.1f}K"
        else:
            return str(int(volume))
    
    def calculate_technical_indicators(self, price_data):
        """Calculate enhanced technical indicators"""
        if len(price_data) < 20:
            return {
                'sma_20': 0,
                'sma_50': 0,
                'rsi': 50,
                'volatility': 0,
                'price_momentum': 0
            }
        
        closes = [p['close'] for p in price_data]
        
        # Moving averages
        sma_20 = sum(closes[-20:]) / 20
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
        
        # RSI calculation (simplified)
        gains = []
        losses = []
        for i in range(1, min(15, len(closes))):
            change = closes[-i] - closes[-i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Volatility
        mean_price = sum(closes[-20:]) / 20
        variance = sum((x - mean_price) ** 2 for x in closes[-20:]) / 20
        volatility = variance ** 0.5
        
        # Price momentum
        current_price = closes[-1]
        price_momentum = ((current_price - sma_20) / sma_20) * 100
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'volatility': volatility,
            'price_momentum': price_momentum
        }
    
    def determine_enhanced_rating(self, stock_info):
        """Enhanced rating system with multiple factors"""
        try:
            score = 0
            
            # Price performance factors
            weekly_change = stock_info.get('weekly_change', 0)
            monthly_change = stock_info.get('monthly_change', 0)
            
            if weekly_change > 5:
                score += 2
            elif weekly_change > 0:
                score += 1
            elif weekly_change < -5:
                score -= 2
            
            if monthly_change > 10:
                score += 1
            elif monthly_change < -10:
                score -= 1
            
            # Valuation factors
            pe_ratio = stock_info.get('trailingPE', 0)
            if pe_ratio and 15 < pe_ratio < 25:
                score += 1
            elif pe_ratio and pe_ratio > 40:
                score -= 1
            
            # Technical factors
            rsi = stock_info.get('rsi', 50)
            if 30 < rsi < 70:
                score += 1
            elif rsi < 30:
                score += 2  # Oversold - potential buy
            elif rsi > 70:
                score -= 1  # Overbought
            
            # Momentum factors
            price_momentum = stock_info.get('price_momentum', 0)
            if price_momentum > 5:
                score += 1
            elif price_momentum < -5:
                score -= 1
            
            # Rating determination
            if score >= 4:
                return 'STRONG BUY'
            elif score >= 2:
                return 'BUY'
            elif score >= 0:
                return 'HOLD'
            elif score >= -2:
                return 'SELL'
            else:
                return 'STRONG SELL'
                
        except Exception:
            return 'HOLD'
    
    def collect_dynamic_data(self, user_query: str):
        """Main function with enhanced query processing and parallel data collection"""
        start_time = datetime.now()
        logging.info(f"Processing enhanced query: {user_query}")
        
        # Enhanced query analysis
        query_analysis = self.query_analyzer.analyze_query(user_query)
        logging.info(f"Enhanced query analysis: {query_analysis}")
        
        # Discover tickers with confidence-based selection
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
        
        # Parallel data collection
        parallel_results = self.fetch_parallel_stock_data(tickers, period='6mo', interval='1d')
        
        # Enrich data
        enriched_fundamentals = self.enrich_fundamentals_data(
            parallel_results['fundamentals'], 
            parallel_results['price_data']
        )
        
        # Calculate total execution time
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

# Initialize global collector
parallel_financial_collector = ParallelFinancialDataCollector()

# Maintain backward compatibility
def get_stock_analysis(query):
    """Enhanced main function for stock analysis"""
    return parallel_financial_collector.collect_dynamic_data(query)

def get_sector_data(tickers, period='6mo', interval='1d'):
    """Enhanced legacy function with parallel processing"""
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

# For backward compatibility
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
