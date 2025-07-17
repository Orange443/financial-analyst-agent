# report_generator.py
import matplotlib
matplotlib.use('Agg')

import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Frame, PageTemplate
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def header_footer(canvas, doc):
    canvas.saveState()
    styles = getSampleStyleSheet()
    
    # Header
    header = Paragraph("Confidential Financial Analysis", styles['Normal'])
    w, h = header.wrap(doc.width, doc.topMargin)
    header.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h)

    # Footer
    footer = Paragraph(f"Page {doc.page} | Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal'])
    w, h = footer.wrap(doc.width, doc.bottomMargin)
    footer.drawOn(canvas, doc.leftMargin, h)
    
    canvas.restoreState()

def create_stock_chart(stock_data, output_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 4))
    for ticker, data_points in stock_data.items():
        if isinstance(data_points, list) and all('date' in d and 'close' in d for d in data_points):
            dates = [datetime.strptime(dp['date'], '%Y-%m-%d') for dp in data_points]
            closes = [dp['close'] for dp in data_points]
            ax.plot(dates, closes, marker='o', linestyle='-', label=ticker)
    ax.set_title("Weekly Stock Performance", fontsize=14, weight='bold')
    ax.set_ylabel("Closing Price", fontsize=10)
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.close(fig)

# In report_generator.py

def generate_enhanced_pdf_report(report_title: str, report_data: dict, stock_data: dict) -> str:
    os.makedirs('static', exist_ok=True)
    pdf_filename = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join("static", pdf_filename)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    styles = getSampleStyleSheet()
    
    # Check if styles exist before adding them to prevent errors on subsequent runs
    if 'ReportTitle' not in styles:
        styles.add(ParagraphStyle(name='ReportTitle', fontSize=22, alignment=TA_CENTER, spaceAfter=20, textColor=colors.HexColor("#1A237E"), fontName='Helvetica-Bold'))
    if 'SectionTitle' not in styles:
        styles.add(ParagraphStyle(name='SectionTitle', fontSize=16, spaceAfter=12, textColor=colors.HexColor("#004D40"), fontName='Helvetica-Bold'))
    # FIX: Add the same check for the 'BodyText' style
    if 'BodyText' not in styles:
        styles.add(ParagraphStyle(name='BodyText', fontSize=11, leading=14, alignment=TA_LEFT))

    story = []
    story.append(Paragraph(f"Weekly Report: {report_title.title()}", styles['ReportTitle']))
    
    story.append(Paragraph("Executive Summary", styles['SectionTitle']))
    story.append(Paragraph(report_data['executive_summary'], styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Detailed Analysis", styles['SectionTitle']))
    story.append(Paragraph(report_data['detailed_analysis'], styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("Expert Outlook", styles['SectionTitle']))
    story.append(Paragraph(report_data['expert_outlook'], styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())
    
    story.append(Paragraph("Performance Chart", styles['SectionTitle']))
    chart_path = os.path.join("static", "stock_chart.png")
    create_stock_chart(stock_data, chart_path)
    story.append(Image(chart_path, width=7*inch, height=2.8*inch))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return pdf_filename