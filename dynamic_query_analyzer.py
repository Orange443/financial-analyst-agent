# dynamic_query_analyzer.py
from typing import List, Dict

class EnhancedDynamicQueryAnalyzer:
    def __init__(self):
        self.sector_keywords = {
            'fmcg': ['consumer', 'fmcg', 'goods', 'brands'],
            'technology': ['tech', 'it', 'software', 'computer'],
            'banking': ['bank', 'finance', 'financial'],
            'pharmaceutical': ['pharma', 'drug', 'medicine'],
            'automotive': ['auto', 'car', 'vehicle'],
            'energy': ['oil', 'gas', 'energy']
        }
        
        self.nse_sector_stocks = {
            'fmcg': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO', 'EMAMILTD', 'TATACONSUM'],
            'technology': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
            'banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK'],
            'pharmaceutical': ['SUNPHARMA', 'DRREDDY', 'CIPLA'],
            'automotive': ['MARUTI', 'TATAMOTORS', 'M&M'],
            'energy': ['RELIANCE', 'ONGC', 'IOC']
        }
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine sector"""
        query_lower = query.lower()
        
        for sector, keywords in self.sector_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    'type': 'sector',
                    'sector': sector,
                    'confidence': 1.0
                }
        
        return {
            'type': 'general',
            'sector': 'fmcg',
            'confidence': 0.5
        }
    
    def discover_tickers(self, query_analysis: Dict) -> List[str]:
        """Discover relevant tickers"""
        sector = query_analysis.get('sector', 'fmcg')
        stocks = self.nse_sector_stocks.get(sector, self.nse_sector_stocks['fmcg'])
        return [f"{stock}.NS" for stock in stocks]
