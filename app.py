# app.py - Clean version without API monitoring
from flask import Flask, render_template, request, Response, send_from_directory, jsonify
import os
import json
import base64
import time
import random
import logging
from datetime import datetime
from dotenv import load_dotenv

from tools.yfinance_tools import get_stock_analysis
from tools.google_tools import DynamicNewsCollector
from report_generator import generate_investment_report
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
app = Flask(__name__)

# Simple logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_app.log'),
        logging.StreamHandler()
    ]
)

def generate_dynamic_image_prompt(query_analysis, fundamentals_data):
    """Generate dynamic image prompts based on sector analysis and performance data"""
    
    sector_name = query_analysis.get('sector', 'general')
    query_type = query_analysis.get('type', 'general')
    confidence = query_analysis.get('confidence', 0.5)
    
    # Calculate performance sentiment from actual data
    sentiment_data = calculate_performance_sentiment(fundamentals_data)
    
    # Base industry context mapping
    industry_contexts = {
        'fmcg': 'consumer goods, retail, packaged products, everyday essentials',
        'banking': 'financial services, banking facilities, monetary systems',
        'technology': 'IT services, software development, digital infrastructure',
        'pharmaceutical': 'healthcare, medical research, drug development',
        'automotive': 'vehicle manufacturing, transportation, mobility',
        'energy': 'power generation, energy infrastructure, utilities',
        'realty': 'real estate, construction, property development',
        'telecom': 'telecommunications, network infrastructure, connectivity',
        'metals': 'metal production, mining, industrial materials',
        'cement': 'construction materials, building infrastructure'
    }
    
    # Get industry context
    industry_context = industry_contexts.get(sector_name, f"{sector_name} industry operations and facilities")
    
    # Performance-based visual elements
    performance_elements = generate_performance_visual_elements(sentiment_data)
    
    # Market condition descriptors
    market_conditions = get_market_condition_descriptors(sentiment_data)
    
    # Construct dynamic prompt
    if query_type == 'sector':
        prompt = f"""Create a professional photograph showing {industry_context} in a modern, active business environment. 
        
Scene should convey: {market_conditions['mood']} market sentiment with {market_conditions['activity_level']} business activity.
        
Visual elements to include:
- Modern facilities and equipment related to {sector_name}
- {performance_elements['visual_mood']}
- {performance_elements['color_scheme']}
- Professional workers in a contemporary setting
- Clean, well-lit corporate environment
        
Style: High-quality corporate photography, professional lighting, modern industrial aesthetic.
Exclude: Any text, logos, or graphic overlays.
Quality: Sharp, detailed, suitable for business presentation."""
        
    elif query_type == 'company':
        company_name = query_analysis.get('company', 'corporate')
        prompt = f"""Professional photograph of a modern {company_name} business facility or office environment.
        
Scene characteristics:
- Contemporary corporate workspace
- {market_conditions['mood']} business atmosphere
- {performance_elements['visual_mood']}
- Professional employees in modern setting
        
Style: High-quality corporate photography, clean professional aesthetic.
Quality: Sharp, detailed, business-appropriate."""
        
    else:
        prompt = f"""Modern financial business district with professional commercial buildings and active business environment.
        
Atmosphere: {market_conditions['mood']} market sentiment, {market_conditions['activity_level']} business activity.
Visual tone: {performance_elements['color_scheme']}
        
Style: Professional architectural photography, corporate aesthetic."""
    
    # Add confidence indicator
    if confidence > 0.8:
        prompt += f"\nImage should reflect high confidence in {sector_name} sector identification."
    elif confidence > 0.5:
        prompt += f"\nImage should show typical {sector_name} sector business environment."
    else:
        prompt += f"\nImage should show general business environment with {sector_name} sector elements."
    
    logging.info(f"Generated dynamic image prompt for {sector_name}: {prompt[:100]}...")
    return prompt

def calculate_performance_sentiment(fundamentals_data):
    """Calculate overall performance sentiment from actual data"""
    if not fundamentals_data:
        return {
            'overall_sentiment': 'neutral',
            'avg_performance': 0,
            'positive_count': 0,
            'negative_count': 0,
            'volatility_level': 'low'
        }
    
    # Extract performance metrics
    weekly_changes = []
    monthly_changes = []
    volatilities = []
    
    for ticker, data in fundamentals_data.items():
        if isinstance(data, dict):
            weekly_change = data.get('weekly_change', 0)
            monthly_change = data.get('monthly_change', 0)
            volatility = data.get('volatility', 0)
            
            if weekly_change is not None:
                weekly_changes.append(weekly_change)
            if monthly_change is not None:
                monthly_changes.append(monthly_change)
            if volatility is not None:
                volatilities.append(volatility)
    
    # Calculate averages
    avg_weekly = sum(weekly_changes) / len(weekly_changes) if weekly_changes else 0
    avg_monthly = sum(monthly_changes) / len(monthly_changes) if monthly_changes else 0
    avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
    
    # Count positive/negative performers
    positive_count = sum(1 for change in weekly_changes if change > 0)
    negative_count = sum(1 for change in weekly_changes if change < 0)
    
    # Determine overall sentiment
    if avg_weekly > 2:
        sentiment = 'very_positive'
    elif avg_weekly > 0:
        sentiment = 'positive'
    elif avg_weekly > -2:
        sentiment = 'neutral'
    else:
        sentiment = 'negative'
    
    # Determine volatility level
    if avg_volatility > 3:
        volatility_level = 'high'
    elif avg_volatility > 1.5:
        volatility_level = 'moderate'
    else:
        volatility_level = 'low'
    
    return {
        'overall_sentiment': sentiment,
        'avg_performance': avg_weekly,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'total_companies': len(fundamentals_data),
        'volatility_level': volatility_level,
        'avg_monthly': avg_monthly
    }

def generate_performance_visual_elements(sentiment_data):
    """Generate visual elements based on actual performance data"""
    
    sentiment = sentiment_data['overall_sentiment']
    avg_performance = sentiment_data['avg_performance']
    volatility = sentiment_data['volatility_level']
    
    # Dynamic visual mood based on performance
    if sentiment == 'very_positive':
        visual_mood = "bright, optimistic lighting with vibrant energy"
        color_scheme = "warm, energetic tones with bright highlights"
    elif sentiment == 'positive':
        visual_mood = "well-lit, professional atmosphere with positive energy"
        color_scheme = "balanced warm tones with professional lighting"
    elif sentiment == 'neutral':
        visual_mood = "steady, professional lighting with balanced atmosphere"
        color_scheme = "neutral corporate tones with balanced lighting"
    else:  # negative
        visual_mood = "professional but subdued lighting with serious atmosphere"
        color_scheme = "cooler tones with professional, conservative lighting"
    
    # Add volatility-based elements
    if volatility == 'high':
        visual_mood += ", dynamic and active environment"
    elif volatility == 'moderate':
        visual_mood += ", moderately active business environment"
    else:
        visual_mood += ", stable and controlled environment"
    
    return {
        'visual_mood': visual_mood,
        'color_scheme': color_scheme,
        'performance_indicator': f"{avg_performance:+.1f}% weekly performance"
    }

def get_market_condition_descriptors(sentiment_data):
    """Get market condition descriptors based on actual data"""
    
    sentiment = sentiment_data['overall_sentiment']
    positive_ratio = sentiment_data['positive_count'] / max(sentiment_data['total_companies'], 1)
    
    # Mood descriptors
    mood_map = {
        'very_positive': 'optimistic and growth-oriented',
        'positive': 'confident and stable',
        'neutral': 'balanced and professional',
        'negative': 'cautious and conservative'
    }
    
    # Activity level based on positive ratio
    if positive_ratio > 0.7:
        activity_level = 'high'
    elif positive_ratio > 0.4:
        activity_level = 'moderate'
    else:
        activity_level = 'measured'
    
    return {
        'mood': mood_map.get(sentiment, 'professional'),
        'activity_level': activity_level,
        'positive_ratio': positive_ratio
    }

def generate_ai_image(client, image_prompt):
    """Generate AI image"""
    response = client.models.generate_content(
        model="gemini-2.0-flash-preview-image-generation",
        contents=image_prompt,
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )
    return response

def validate_query(query):
    """Validate financial query"""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        prompt = f"Is this query about finance, stocks, companies, or markets? Answer only 'yes' or 'no'.\nQuery: '{query}'"
        response = llm.invoke(prompt)
        
        if isinstance(response, list):
            response_text = str(response[0]).strip().lower()
        else:
            response_text = str(response).strip().lower()
        
        return "yes" in response_text
    except Exception as e:
        logging.warning(f"Query validation failed: {e}")
        return True

def is_query_financial(query: str) -> bool:
    """Enhanced financial query validation"""
    financial_keywords = [
        'stock', 'sector', 'company', 'market', 'investment', 'financial', 'earnings',
        'revenue', 'profit', 'analysis', 'report', 'trading', 'share', 'equity',
        'fmcg', 'banking', 'pharma', 'it', 'tech', 'auto', 'energy'
    ]
    
    query_lower = query.lower()
    
    # Check for financial keywords first
    if any(keyword in query_lower for keyword in financial_keywords):
        return True
    
    # Use LLM validation only if keywords don't match
    return validate_query(query)

@app.route('/')
def index():
    return render_template('index_v3.html')

@app.route('/analyze-stream')
def analyze_stream():
    query = request.args.get('query', 'No query provided')
    
    def generate_updates():
        if not is_query_financial(query):
            yield f"data: {json.dumps({'error': 'Sorry, I can only answer financial queries.'})}\n\n"
            return

        try:
            # Step 1: Data collection
            yield f"data: {json.dumps({'status': 'Analyzing your query and discovering relevant stocks'})}\n\n"
            
            start_time = datetime.now()
            analysis_data = get_stock_analysis(query)
            
            if analysis_data.get('error'):
                yield f"data: {json.dumps({'error': analysis_data['error']})}\n\n"
                return
            
            # Log performance metrics
            if 'performance_metrics' in analysis_data:
                metrics = analysis_data['performance_metrics']
                logging.info(f"Data collection: {metrics['successful_fetches']}/{metrics['companies_processed']} stocks in {metrics['total_execution_time']:.2f}s")

            # Step 2: Dynamic image generation
            yield f"data: {json.dumps({'status': 'Generating intelligent sector visuals based on analysis'})}\n\n"
            
            image_bytes_b64 = ""
            try:
                if GOOGLE_API_KEY:
                    client = genai.Client(api_key=GOOGLE_API_KEY)
                    query_analysis = analysis_data.get('query_analysis', {})
                    fundamentals_data = analysis_data.get('fundamentals', {})
                    
                    # Generate dynamic prompt
                    dynamic_prompt = generate_dynamic_image_prompt(query_analysis, fundamentals_data)
                    
                    logging.info(f"Dynamic prompt generated for {query_analysis.get('sector', 'general')} sector")
                    
                    response = generate_ai_image(client, dynamic_prompt)
                    
                    # Extract image data
                    image_bytes = None
                    candidates = getattr(response, "candidates", None)
                    if candidates and getattr(candidates[0], "content", None):
                        for part in candidates[0].content.parts:
                            if part.inline_data is not None:
                                image_bytes = part.inline_data.data
                                break
                    
                    if image_bytes:
                        image_bytes_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        logging.info("Dynamic sector image generated successfully")
                    else:
                        logging.warning("No image data received from dynamic prompt")
                        
            except Exception as e:
                logging.warning(f"Dynamic image generation failed: {e}")

            # Step 3: Report generation
            yield f"data: {json.dumps({'status': 'Finalizing comprehensive financial report'})}\n\n"
            
            try:
                # Create dynamic report title
                query_analysis = analysis_data.get('query_analysis', {})
                if query_analysis.get('type') == 'sector':
                    report_title = f"{query_analysis['sector'].title()} Sector Analysis"
                elif query_analysis.get('type') == 'company':
                    report_title = f"{query_analysis.get('company', query)} Company Analysis"
                else:
                    report_title = f"Financial Analysis: {query}"
                
                # Ensure static directory exists
                os.makedirs('static', exist_ok=True)
                
                pdf_filename = generate_investment_report(report_title, analysis_data)
                
                # Dynamic Content Generation
                fundamentals = analysis_data.get('fundamentals', {})
                executive_summary = f"This report provides a detailed analysis of the {query_analysis.get('sector', 'general')} sector. "
                if fundamentals:
                    weekly_changes = [d.get('weekly_change', 0) for d in fundamentals.values() if d.get('weekly_change') is not None]
                    if weekly_changes:
                        avg_weekly_change = sum(weekly_changes) / len(weekly_changes)
                        executive_summary += f"The sector shows an average weekly change of {avg_weekly_change:.2f}%. "

                # Key Findings
                key_findings = {
                    'top_performer': 'N/A',
                    'top_performer_change': 0,
                    'lagging_stock': 'N/A',
                    'lagging_stock_change': 0,
                    'highest_volume_stock': 'N/A',
                    'overall_sentiment': 'Neutral'
                }

                if fundamentals:
                    # Top Performer
                    top_performer = max(fundamentals.values(), key=lambda x: x.get('weekly_change', -1000))
                    key_findings['top_performer'] = top_performer.get('name', 'N/A')
                    key_findings['top_performer_change'] = top_performer.get('weekly_change', 0)

                    # Lagging Stock
                    lagging_stock = min(fundamentals.values(), key=lambda x: x.get('weekly_change', 1000))
                    key_findings['lagging_stock'] = lagging_stock.get('name', 'N/A')
                    key_findings['lagging_stock_change'] = lagging_stock.get('weekly_change', 0)

                    # Highest Volume
                    highest_volume_stock = max(fundamentals.values(), key=lambda x: x.get('marketCap', 0))
                    key_findings['highest_volume_stock'] = highest_volume_stock.get('name', 'N/A')

                    # Overall sentiment
                    sentiment_data = calculate_performance_sentiment(fundamentals)
                    key_findings['overall_sentiment'] = sentiment_data['overall_sentiment'].title()

                # Recommendations
                recommendations = []
                if fundamentals:
                    if key_findings['top_performer_change'] > 5:
                        recommendations.append(f"Consider focusing on {key_findings['top_performer']} due to its strong weekly performance.")
                    if key_findings['lagging_stock_change'] < -5:
                        recommendations.append(f"Exercise caution with {key_findings['lagging_stock']} due to its recent underperformance.")
                    
                    # Add sentiment-based recommendations
                    sentiment_data = calculate_performance_sentiment(fundamentals)
                    if sentiment_data['overall_sentiment'] == 'very_positive':
                        recommendations.append("Strong sector momentum suggests maintaining overweight position.")
                    elif sentiment_data['overall_sentiment'] == 'negative':
                        recommendations.append("Consider defensive positioning given sector headwinds.")

                final_payload = {
                    'done': True,
                    'query_analysis': analysis_data.get('query_analysis', {}),
                    'companies_analyzed': len(analysis_data.get('tickers', [])),
                    'successful_tickers': len([t for t in analysis_data.get('fundamentals', {})
                                             if analysis_data['fundamentals'][t].get('regularMarketPrice')]),
                    'performance_metrics': analysis_data.get('performance_metrics', {}),
                    'cover_image_b64': image_bytes_b64,
                    'pdf_path': f"/static/{pdf_filename}",
                    'report': {
                        'title': report_title,
                        'type': query_analysis.get('type', 'general'),
                        'sector': query_analysis.get('sector', 'Mixed'),
                        'confidence': query_analysis.get('confidence', 0.5),
                        'executive_summary': executive_summary,
                        'key_findings': key_findings,
                        'recommendations': recommendations
                    }
                }
                yield f"data: {json.dumps(final_payload)}\n\n"

            except Exception as e:
                logging.error(f"Report generation failed: {e}")
                yield f"data: {json.dumps({'error': f'Report generation failed: {str(e)}'})}\n\n"

        except Exception as e:
            logging.error(f"Critical error in analyze_stream: {e}")
            yield f"data: {json.dumps({'error': f'Server error: {str(e)}. Please try again.'})}\n\n"

    return Response(generate_updates(), mimetype='text/event-stream')

@app.errorhandler(500)
def handle_internal_error(error):
    logging.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal error occurred. Please try again.',
        'retry_suggested': True
    }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    logging.info("Starting Financial Analysis Application")
    app.run(debug=True, port=5001)
