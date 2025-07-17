# app.py
from flask import Flask, render_template, request, Response, send_from_directory
import os
import json
import time
import base64
from dotenv import load_dotenv

from report_generator import generate_enhanced_pdf_report
from agents.report_agent import app as financial_agent_app
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
app = Flask(__name__)

def is_query_financial(query: str) -> bool:
    print(f"Validating query: '{query}'")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
        prompt = f"Is the following query about finance, stocks, or companies? Answer only 'yes' or 'no'.\nQuery: \"{query}\""
        response = llm.invoke(prompt)
        # If response is a list, get the first string element
        if isinstance(response, list):
            response_text = str(response[0]).strip().lower()
        else:
            response_text = str(response).strip().lower()
        return "yes" in response_text
    except Exception as e:
        print(f"Error during query validation: {e}")
        return True

@app.route('/')
def index():
    return render_template('index_v3.html')

# In app.py
@app.route('/analyze-stream')
def analyze_stream():
    query = request.args.get('query', 'No query provided')
    
    def generate_updates():
        if not is_query_financial(query):
            yield f"data: {json.dumps({'error': 'Sorry, I can only answer financial queries.'})}\n\n"
            return

        try:
            initial_state = {"user_query": query}
            final_state = {}
            
            # --- FIX: The logical flow has been reordered to match the UI ---

            # 1. Start the process
            yield f"data: {json.dumps({'status': 'Parsing your request'})}\n\n"
            time.sleep(0.5)

            # 2. Run the main agent workflow (find stocks, get data, analyze sentiment)
            for event in financial_agent_app.stream(initial_state, stream_mode="values"):
                final_state.update(event) 
                node_name = list(event.keys())[-1]
                
                status_map = {
                    "find_stocks": "Identifying top stocks in sector",
                    "get_market_data": "Fetching market data",
                    "get_news_data": "Gathering news from Google API",
                    "analyze_sentiment": "Analyzing sentiment",
                    # "generate_report" is the last text step, handled after visuals
                }
                if node_name in status_map:
                    yield f"data: {json.dumps({'status': status_map[node_name]})}\n\n"
                    time.sleep(0.5)
            
            # 3. Now, generate the visuals
            yield f"data: {json.dumps({'status': 'Generating report visuals'})}\n\n"
            time.sleep(0.5)
            
            client = genai.Client(api_key=GOOGLE_API_KEY)
            image_prompt = f"A professional, abstract, and visually appealing image for a financial report on '{query}'. Focus on themes of data, growth, and analytics. High-resolution, cinematic quality."
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=image_prompt,
                config=types.GenerateContentConfig(
                  response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            image_bytes = None
            candidates = getattr(response, "candidates", None)
            if candidates and getattr(candidates[0], "content", None) and getattr(candidates[0].content, "parts", None):
                for part in candidates[0].content.parts:
                    if part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        break
            else:
                raise ValueError("Image generation failed, no candidates or parts found.")
            
            if not image_bytes:
                raise ValueError("Image generation failed, no image data found.")

            image_bytes_b64 = base64.b64encode(image_bytes).decode('utf-8')

            # 4. Finalize the report content and generate the PDF
            yield f"data: {json.dumps({'status': 'Finalizing report'})}\n\n"
            time.sleep(0.5)

            # The 'report' data was already generated in the agent stream, now we use it
            report = final_state.get('report')
            if not report:
                raise ValueError("Report generation failed.")

            pdf_filename = generate_enhanced_pdf_report(
                query, report.model_dump(), final_state.get('stock_data', {})
            )
            
            # 5. Send the final payload to the UI
            final_payload = {
                'done': True,
                'report': report.model_dump(),
                'cover_image_b64': image_bytes_b64,
                'pdf_path': f"/static/{pdf_filename}"
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            print(f"An error occurred during streaming: {e}")
            yield f"data: {json.dumps({'error': f'An error occurred: {e}'})}\n\n"

    return Response(generate_updates(), mimetype='text/event-stream')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)