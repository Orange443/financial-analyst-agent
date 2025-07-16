from flask import Flask, render_template, request, send_file
import os
import io
import base64
from dotenv import load_dotenv

# Import reportlab components
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Import Google Generative AI for image generation
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Import the financial analysis agent
from agents.report_agent import app as financial_agent_app

app = Flask(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model for image generation
# Using the specified model for image generation
gemini_image_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-preview-image-generation", google_api_key=GOOGLE_API_KEY)

# Function to generate cover image
def generate_cover_image(query: str) -> str:
    print("Generating cover image...")
    try:
        prompt = f"Generate a visually appealing and professional cover image for a financial analysis report about: {query}. The image should be abstract and convey themes of finance, growth, and data analysis."
        
        # Use the Gemini model to generate the image
        response = gemini_image_model.invoke([HumanMessage(content=prompt)])
        
        # Assuming the response content is a base64 encoded string
        image_data_base64 = response.content
        
        # Decode the base64 string
        image_bytes = base64.b64decode(image_data_base64)
        
        image_path = os.path.join("static", "cover_image.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        print(f"Cover image generated and saved to {image_path}")
        return image_path
    except Exception as e:
        print(f"Error generating cover image: {e}")
        # Fallback to a dummy image if real generation fails
        from PIL import Image as PILImage
        img = PILImage.new('RGB', (600, 400), color = 'gray')
        image_path = os.path.join("static", "cover_image_dummy.png")
        img.save(image_path)
        print(f"Using dummy cover image: {image_path}")
        return image_path

# Function to generate PDF report
def generate_pdf_report(analysis_text: str, image_path: str) -> str:
    print("Generating PDF report...")
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add cover image if available
    if image_path and os.path.exists(image_path):
        img = Image(image_path)
        img.width = 6 * inch  # Adjust width as needed
        img.height = 4 * inch # Adjust height as needed
        story.append(img)
        story.append(Spacer(1, 0.5 * inch))
        story.append(Paragraph("Financial Analysis Report", styles['h1']))
        story.append(PageBreak())

    # Add analysis summary
    story.append(Paragraph("Detailed Analysis", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))
    for paragraph in analysis_text.split('\n\n'): # Split by double newline for paragraphs
        story.append(Paragraph(paragraph, styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    pdf_buffer.seek(0)

    pdf_filename = "financial_report.pdf"
    with open(os.path.join("static", pdf_filename), "wb") as f:
        f.write(pdf_buffer.read())
    print(f"PDF report generated and saved to static/{pdf_filename}")
    return pdf_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form['query']
    
    # Run the financial analysis agent
    print(f"Running financial agent for query: {query}")
    initial_state = {"user_query": query, "stocks": [], "stock_data": {}, "news_data": {}, "analysis": "", "error": None}
    final_state = financial_agent_app.invoke(initial_state)
    analysis_result = final_state.get('analysis', 'No analysis generated.')
    
    # Generate cover image
    # Ensure the 'static' directory exists
    os.makedirs('static', exist_ok=True)
    cover_image_path = generate_cover_image(query)
    
    # Generate PDF report
    pdf_filename = generate_pdf_report(analysis_result, cover_image_path)
    
    return render_template('result.html', analysis_result=analysis_result, pdf_path=f"/static/{pdf_filename}")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the Flask app.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on.')
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
