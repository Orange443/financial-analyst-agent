# report_generator.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import logging

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# Configure logging
logging.basicConfig(level=logging.INFO)

def header_footer(canvas, doc):
    """Professional header/footer for investment reports"""
    canvas.saveState()
    
    # Header
    header = Paragraph(
        "<b>CONFIDENTIAL</b> | Investment Research Report",
        getSampleStyleSheet()['Normal']
    )
    w, h = header.wrap(doc.width, doc.topMargin)
    header.drawOn(canvas, doc.leftMargin, letter[1] - doc.topMargin + 10)
    
    # Footer
    footer = Paragraph(
        f"Page {canvas.getPageNumber()} | Generated on {datetime.now().strftime('%B %d, %Y')} | For Institutional Use Only",
        getSampleStyleSheet()['Normal']
    )
    w, h = footer.wrap(doc.width, doc.bottomMargin)
    footer.drawOn(canvas, doc.leftMargin, h)
    
    canvas.restoreState()

def safe_format_number(value, format_type="float", decimals=2):
    """Safely format numbers with fallback for None/invalid values"""
    if value is None or value == 0:
        return "–"
    
    try:
        if format_type == "currency":
            return f"₹{float(value):,.0f}"
        elif format_type == "percentage":
            return f"{float(value):+.{decimals}f}%"
        elif format_type == "crores":
            return f"₹{float(value)/10000000:,.0f}Cr"
        elif format_type == "float":
            return f"{float(value):.{decimals}f}"
        else:
            return str(value)
    except (ValueError, TypeError):
        return "–"

def create_comprehensive_price_chart(stock_data, price_targets, output_path):
    """Create enhanced price chart with targets and volume"""
    try:
        # Create figure with subplots for price and volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        has_data = False
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        for i, (ticker, data_points) in enumerate(stock_data.items()):
            if not data_points or len(data_points) < 2:
                continue
                
            dates = []
            closes = []
            volumes = []
            
            for dp in data_points:
                try:
                    if isinstance(dp, dict) and 'date' in dp and 'close' in dp:
                        dates.append(datetime.strptime(dp['date'], '%Y-%m-%d'))
                        closes.append(float(dp['close']))
                        volumes.append(float(dp.get('volume', 0)))
                except (ValueError, TypeError):
                    continue
            
            if len(dates) > 1:
                color = colors_list[i % len(colors_list)]
                
                # Price chart
                ax1.plot(dates, closes, marker='o', linestyle='-', 
                        linewidth=2.5, label=ticker.replace('.NS', ''), color=color, markersize=4)
                
                # Volume chart
                ax2.bar(dates, volumes, alpha=0.6, color=color, width=1.5)
                
                # Price target line
                if price_targets and ticker in price_targets:
                    target = price_targets[ticker]
                    ax1.axhline(y=target, color=color, linestyle='--', 
                              alpha=0.8, linewidth=2)
                    
                    # Add target label
                    if dates:
                        ax1.text(dates[-1], target, f'  Target: ₹{target:,.0f}', 
                                fontsize=9, ha='left', va='bottom', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
                
                has_data = True
        
        if not has_data:
            ax1.text(0.5, 0.5, 'No price data available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, style='italic')
        
        # Price chart formatting
        ax1.set_title("Stock Price Performance vs. Price Targets", weight='bold', fontsize=16, pad=20)
        ax1.set_ylabel("Price (₹)", fontsize=13, weight='bold')
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(fontsize=10, loc='upper left', frameon=True, shadow=True)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Volume chart formatting
        ax2.set_ylabel("Volume", fontsize=11, weight='bold')
        ax2.set_xlabel("Date", fontsize=13, weight='bold')
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.tick_params(axis='y', labelsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis for volume
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        
        plt.tight_layout(pad=3.0)
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Error creating price chart: {e}")
        # Create fallback chart
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Chart generation failed: Technical issue', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Stock Price Performance", weight='bold', fontsize=16)
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_performance_comparison_chart(fundamentals, output_path):
    """Create comprehensive performance comparison chart"""
    try:
        if not fundamentals:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No fundamental data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Comparison', weight='bold', fontsize=16)
            plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            return

        stocks = [ticker.replace('.NS', '') for ticker in fundamentals.keys()]
        x_pos = np.arange(len(stocks))
        
        # Extract performance data
        weekly_changes = [fundamentals[list(fundamentals.keys())[i]].get('weekly_change', 0) for i in range(len(stocks))]
        monthly_changes = [fundamentals[list(fundamentals.keys())[i]].get('monthly_change', 0) for i in range(len(stocks))]
        quarterly_changes = [fundamentals[list(fundamentals.keys())[i]].get('quarterly_change', 0) for i in range(len(stocks))]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        width = 0.25
        
        # Create grouped bar chart with enhanced styling
        bars1 = ax.bar(x_pos - width, weekly_changes, width, label='1 Week', 
                      color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x_pos, monthly_changes, width, label='1 Month', 
                      color='#4169E1', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x_pos + width, quarterly_changes, width, label='3 Months', 
                      color='#DC143C', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        def add_value_labels(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if abs(height) > 0.1:  # Only show labels for significant values
                    ax.annotate(f'{value:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3 if height >= 0 else -15),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top',
                               fontsize=8, weight='bold')
        
        add_value_labels(bars1, weekly_changes)
        add_value_labels(bars2, monthly_changes)
        add_value_labels(bars3, quarterly_changes)
        
        # Chart formatting
        ax.set_title('Performance Comparison Across Time Periods', weight='bold', fontsize=16, pad=20)
        ax.set_ylabel('Returns (%)', fontsize=13, weight='bold')
        ax.set_xlabel('Companies', fontsize=13, weight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stocks, rotation=45, ha='right')
        ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.tick_params(axis='both', labelsize=10)
        
        # Add background colors for positive/negative regions
        ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color='green')
        ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color='red')
        
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Error creating performance chart: {e}")
        # Create fallback chart
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Chart generation failed: Technical issue', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Performance Comparison', weight='bold', fontsize=16)
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def create_sector_overview_chart(fundamentals, output_path):
    """Create sector overview with key metrics"""
    try:
        if not fundamentals:
            return
        
        # Extract key metrics
        companies = [info.get('name', ticker)[:15] for ticker, info in fundamentals.items()]
        market_caps = [info.get('marketCap', 0)/10000000 for info in fundamentals.values()]  # In crores
        pe_ratios = [info.get('trailingPE', 0) for info in fundamentals.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Market Cap Chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(companies)))
        wedges, texts, autotexts = ax1.pie(market_caps, labels=companies, autopct='%1.1f%%', 
                                          colors=colors, startangle=90, textprops={'fontsize': 9})
        ax1.set_title('Market Capitalization Distribution', weight='bold', fontsize=14, pad=20)
        
        # P/E Ratio Chart
        bars = ax2.bar(range(len(companies)), pe_ratios, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('P/E Ratio Comparison', weight='bold', fontsize=14, pad=20)
        ax2.set_ylabel('P/E Ratio', fontsize=12, weight='bold')
        ax2.set_xlabel('Companies', fontsize=12, weight='bold')
        ax2.set_xticks(range(len(companies)))
        ax2.set_xticklabels(companies, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, pe_ratios)):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
    except Exception as e:
        logging.error(f"Error creating sector overview: {e}")

def generate_investment_report(report_title, analysis_data):
    """
    Generate comprehensive investment report with enhanced formatting and charts
    """
    
    try:
        # Extract data from analysis
        stock_data = analysis_data.get('price_data', {})
        sector_fundamentals = analysis_data.get('fundamentals', {})
        errors = analysis_data.get('errors', {})
        query_analysis = analysis_data.get('query_analysis', {})
        performance_metrics = analysis_data.get('performance_metrics', {})
        
        logging.info(f"Generating report for {len(sector_fundamentals)} companies")
        
        # Create simple price targets
        price_targets = {}
        for ticker, info in sector_fundamentals.items():
            current_price = info.get('regularMarketPrice', 0)
            if current_price and current_price > 0:
                weekly_change = info.get('weekly_change', 0)
                if weekly_change > 2:
                    target = current_price * 1.15  # 15% upside for strong performers
                elif weekly_change > 0:
                    target = current_price * 1.10  # 10% upside for moderate performers
                else:
                    target = current_price * 1.05  # 5% upside for others
                price_targets[ticker] = target
        
        # Generate the PDF report
        os.makedirs('static', exist_ok=True)
        pdf_filename = f"investment_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join("static", pdf_filename)
        
        # Create document with enhanced margins
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, 
                               topMargin=1.2*inch, bottomMargin=1*inch, 
                               leftMargin=0.8*inch, rightMargin=0.8*inch)
        styles = getSampleStyleSheet()
        
        # Enhanced Custom Styles
        custom_styles = {
            'ReportTitle': ParagraphStyle(
                name='ReportTitle', fontSize=32, fontName='Helvetica-Bold',
                alignment=TA_CENTER, textColor=colors.HexColor("#1A365D"),
                spaceAfter=25, spaceBefore=20, leading=36
            ),
            'SubTitle': ParagraphStyle(
                name='SubTitle', fontSize=20, fontName='Helvetica-Bold',
                alignment=TA_CENTER, textColor=colors.HexColor("#2D5F7F"),
                spaceAfter=20, leading=24
            ),
            'SectionTitle': ParagraphStyle(
                name='SectionTitle', fontSize=18, fontName='Helvetica-Bold',
                spaceAfter=12, textColor=colors.HexColor("#2C5530"),
                leading=22, borderWidth=2, borderColor=colors.HexColor("#2C5530"),
                borderPadding=6, backColor=colors.HexColor("#F0F8F0")
            ),
            'SubSectionTitle': ParagraphStyle(
                name='SubSectionTitle', fontSize=15, fontName='Helvetica-Bold',
                spaceAfter=10, textColor=colors.HexColor("#1A472A"),
                leading=18
            ),
            'BodyText': ParagraphStyle(
                name='BodyText', fontSize=12, leading=18, alignment=TA_JUSTIFY,
                spaceAfter=10, fontName='Helvetica'
            ),
            'BulletText': ParagraphStyle(
                name='BulletText', fontSize=11, leading=16, alignment=TA_LEFT,
                leftIndent=25, spaceAfter=6, fontName='Helvetica'
            ),
            'HighlightBox': ParagraphStyle(
                name='HighlightBox', fontSize=13, alignment=TA_LEFT,
                backColor=colors.HexColor("#E6F7FF"), leftIndent=20, rightIndent=20,
                borderPadding=15, spaceBefore=12, spaceAfter=12, borderWidth=2,
                borderColor=colors.HexColor("#1890FF"), leading=20
            ),
            'MetricBox': ParagraphStyle(
                name='MetricBox', fontSize=12, alignment=TA_CENTER,
                backColor=colors.HexColor("#F6F8FA"), borderPadding=12,
                spaceBefore=8, spaceAfter=8, borderWidth=1,
                borderColor=colors.HexColor("#D1D5DA"), leading=16
            ),
            'Disclaimer': ParagraphStyle(
                name='Disclaimer', fontSize=10, alignment=TA_JUSTIFY,
                textColor=colors.grey, leading=14, fontName='Helvetica'
            )
        }
        
        # Add styles to stylesheet
        for name, style in custom_styles.items():
            if name not in styles:
                styles.add(style)
        
        story = []
        
        # === COVER PAGE ===
        story.append(Paragraph("INVESTMENT RESEARCH REPORT", styles['ReportTitle']))
        story.append(Spacer(1, 0.4*inch))
        story.append(Paragraph(f"{report_title}", styles['SubTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Enhanced Investment Summary Table
        summary_data = []
        
        # Calculate aggregate metrics
        total_companies = len(sector_fundamentals)
        avg_pe = np.mean([info.get('trailingPE', 0) for info in sector_fundamentals.values() if info.get('trailingPE')])
        avg_weekly_change = np.mean([info.get('weekly_change', 0) for info in sector_fundamentals.values()])
        
        investment_summary = [
            ["Analysis Overview", "Details"],
            ["Report Type", query_analysis.get('type', 'Sector').title() + " Analysis"],
            ["Primary Focus", query_analysis.get('sector', 'Mixed Market').title()],
            ["Analysis Confidence", f"{query_analysis.get('confidence', 0.5)*100:.0f}%"],
            ["Companies Analyzed", str(total_companies)],
            ["Successful Data Fetch", str(len([t for t in sector_fundamentals if sector_fundamentals[t].get('regularMarketPrice')]))],
            ["Average P/E Ratio", f"{avg_pe:.1f}" if avg_pe else "N/A"],
            ["Average Weekly Performance", f"{avg_weekly_change:+.2f}%"],
            ["Analysis Period", "6 Months Historical Data"],
            ["Report Generation", datetime.now().strftime('%B %d, %Y at %I:%M %p')]
        ]
        
        invest_table = Table(investment_summary, colWidths=[2.5*inch, 2.5*inch])
        invest_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 14),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 11),
            ('GRID', (0,0), (-1,-1), 1.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8F9FA"), colors.HexColor("#E9ECEF")]),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8)
        ]))
        
        story.append(invest_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Performance Summary Box
        if performance_metrics:
            perf_text = f"<b>System Performance:</b> Analyzed {performance_metrics.get('companies_processed', 0)} companies in {performance_metrics.get('total_execution_time', 0):.2f} seconds with {performance_metrics.get('success_rate', 0)*100:.0f}% success rate."
            story.append(Paragraph(perf_text, styles['MetricBox']))
        
        story.append(PageBreak())
        
        # === EXECUTIVE SUMMARY ===
        story.append(Paragraph("EXECUTIVE SUMMARY", styles['SectionTitle']))
        
        story.append(Spacer(1, 0.1*inch))
        
        # Generate dynamic executive summary
        executive_summary = f"This comprehensive analysis covers {total_companies} companies "
        if query_analysis.get('type') == 'sector':
            executive_summary += f"in the {query_analysis['sector']} sector "
        
        executive_summary += f"with {query_analysis.get('confidence', 0.5)*100:.0f}% confidence in identification. "
        
        # Add performance insights
        if sector_fundamentals:
            performers = [info.get('weekly_change', 0) for info in sector_fundamentals.values()]
            avg_performance = sum(performers) / len(performers)
            positive_performers = sum(1 for p in performers if p > 0)
            
            if avg_performance > 3:
                executive_summary += f"The analysis reveals strong positive momentum with an average return of {avg_performance:+.1f}%. "
            elif avg_performance < -3:
                executive_summary += f"The sector faces challenges with an average decline of {avg_performance:.1f}%. "
            else:
                executive_summary += f"Mixed performance observed with {positive_performers} out of {len(performers)} companies showing positive returns. "
        
        executive_summary += "Detailed analysis, technical indicators, and investment recommendations are provided in the following sections."
        
        story.append(Paragraph(executive_summary, styles['HighlightBox']))
        story.append(Spacer(1, 0.2*inch))
        
        # === FINANCIAL METRICS & VALUATION ===
        story.append(Paragraph("COMPREHENSIVE FINANCIAL METRICS", styles['SectionTitle']))
        
        # Enhanced metrics table with more columns
        metrics_table = [
            ["Company", "Price (₹)", "Mkt Cap", "P/E", "P/B", "1W %", "1M %", "Volume", "Rating"]
        ]
        
        if sector_fundamentals:
            for symbol, info in sector_fundamentals.items():
                metrics_table.append([
                    info.get("name", symbol)[:18],
                    safe_format_number(info.get('regularMarketPrice'), "currency"),
                    safe_format_number(info.get('marketCap'), "crores"),
                    safe_format_number(info.get('trailingPE'), "float", 1),
                    safe_format_number(info.get('priceToBook'), "float", 1),
                    safe_format_number(info.get('weekly_change'), "percentage", 1),
                    safe_format_number(info.get('monthly_change'), "percentage", 1),
                    info.get('volume_formatted', '–'),
                    info.get('rating', 'HOLD')
                ])
        else:
            metrics_table.append(["No data available", "–", "–", "–", "–", "–", "–", "–", "–"])
        
        metrics_table_obj = Table(metrics_table, repeatRows=1)
        metrics_table_obj.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2C5530")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#2C5530")),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F8F9FA"), colors.HexColor("#E9ECEF")]),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6)
        ]))
        
        story.append(metrics_table_obj)
        story.append(Spacer(1, 0.3*inch))
        
        # === VISUAL ANALYSIS ===
        story.append(Paragraph("TECHNICAL & PERFORMANCE ANALYSIS", styles['SectionTitle']))
        
        # Generate and include charts
        if stock_data:
            # Comprehensive price chart
            chart_path1 = os.path.join("static", "comprehensive_price_chart.png")
            create_comprehensive_price_chart(stock_data, price_targets, chart_path1)
            story.append(Image(chart_path1, width=7.5*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))
        
        if sector_fundamentals:
            # Performance comparison chart
            chart_path2 = os.path.join("static", "performance_comparison.png")
            create_performance_comparison_chart(sector_fundamentals, chart_path2)
            story.append(Image(chart_path2, width=7.5*inch, height=4*inch))
            story.append(Spacer(1, 0.2*inch))
            
            # Sector overview chart
            chart_path3 = os.path.join("static", "sector_overview.png")
            create_sector_overview_chart(sector_fundamentals, chart_path3)
            story.append(Image(chart_path3, width=7.5*inch, height=4*inch))
            story.append(Spacer(1, 0.3*inch))
        
        # === INVESTMENT RECOMMENDATIONS ===
        story.append(Paragraph("INVESTMENT RECOMMENDATIONS", styles['SectionTitle']))
        
        # Generate intelligent recommendations
        recommendations = []
        if sector_fundamentals:
            # Identify top and bottom performers
            performers = [(ticker, info.get('weekly_change', 0), info.get('name', ticker)) 
                         for ticker, info in sector_fundamentals.items()]
            performers.sort(key=lambda x: x[1], reverse=True)
            
            if performers:
                top_performer = performers[0]
                bottom_performer = performers[-1]
                
                recommendations.append(f"<b>Top Performer:</b> {top_performer[2]} showing {top_performer[1]:+.1f}% weekly return demonstrates strong momentum")
                
                if bottom_performer[1] < -2:
                    recommendations.append(f"<b>Monitor Closely:</b> {bottom_performer[2]} with {bottom_performer[1]:.1f}% decline requires attention")
                
                # Overall sector recommendation
                avg_performance = sum(p[1] for p in performers) / len(performers)
                if avg_performance > 2:
                    recommendations.append("<b>Sector Outlook:</b> Positive momentum suggests overweight positioning appropriate")
                elif avg_performance < -2:
                    recommendations.append("<b>Sector Caution:</b> Defensive positioning recommended given sector headwinds")
                else:
                    recommendations.append("<b>Selective Strategy:</b> Stock-specific analysis recommended given mixed performance")
                
                # Valuation-based recommendations
                high_pe_stocks = [p for p in performers if sector_fundamentals[p[0]].get('trailingPE', 0) > 30]
                if high_pe_stocks:
                    recommendations.append(f"<b>Valuation Alert:</b> {len(high_pe_stocks)} stocks trading at elevated P/E ratios above 30x")
        
        if recommendations:
            recs = [
                ListItem(Paragraph(f"• {rec}", styles['BulletText']), bulletColor='#2C5530')
                for rec in recommendations
            ]
            story.append(ListFlowable(recs, bulletType='bullet', bulletFontSize=11))
        else:
            story.append(Paragraph("Detailed recommendations require additional fundamental analysis beyond current scope.", styles['BodyText']))
        
        story.append(Spacer(1, 0.3*inch))
        
        # === RISK ASSESSMENT ===
        story.append(Paragraph("RISK ASSESSMENT & CONSIDERATIONS", styles['SectionTitle']))
        
        risk_factors = []
        
        if sector_fundamentals:
            # Market concentration risk
            market_caps = [info.get('marketCap', 0) for info in sector_fundamentals.values() if info.get('marketCap')]
            if market_caps:
                total_market_cap = sum(market_caps)
                largest_weight = max(market_caps) / total_market_cap if total_market_cap > 0 else 0
                if largest_weight > 0.4:
                    risk_factors.append("High concentration risk with largest company representing significant portfolio weight")
            
            # Volatility assessment
            volatilities = [info.get('volatility', 0) for info in sector_fundamentals.values()]
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
            if avg_volatility > 3:
                risk_factors.append("Above-average volatility observed across sector constituents")
            
            # Performance dispersion
            performances = [info.get('weekly_change', 0) for info in sector_fundamentals.values()]
            if performances:
                perf_range = max(performances) - min(performances)
                if perf_range > 10:
                    risk_factors.append("High performance dispersion indicates significant stock-specific risks")
        
        # Standard risks
        risk_factors.extend([
            "Market risk affects all equity investments and cannot be diversified away",
            "Sector-specific regulatory changes may impact all constituents",
            "Economic downturns may disproportionately affect certain sectors",
            "Currency fluctuations may impact companies with international exposure"
        ])
        
        risk_text = " • ".join(risk_factors)
        story.append(Paragraph(f"<b>Key Risk Factors:</b><br/><br/>{risk_text}", styles['BodyText']))
        
        story.append(PageBreak())
        
        # === METHODOLOGY & DISCLAIMERS ===
        story.append(Paragraph("METHODOLOGY & DATA SOURCES", styles['SectionTitle']))
        
        methodology_text = f"""
        <b>Data Collection:</b> This analysis utilizes real-time financial data sourced from Yahoo Finance API, covering {total_companies} companies with 6 months of historical price data.<br/><br/>
        
        <b>Processing Method:</b> Advanced parallel data processing completed analysis in {performance_metrics.get('total_execution_time', 0):.2f} seconds with {performance_metrics.get('success_rate', 0)*100:.0f}% success rate.<br/><br/>
        
        <b>Technical Analysis:</b> Incorporates moving averages, relative strength indicators, and volatility measurements for comprehensive evaluation.<br/><br/>
        
        <b>Confidence Scoring:</b> Multi-factor analysis achieving {query_analysis.get('confidence', 0.5)*100:.0f}% confidence in sector identification through keyword matching, context analysis, and geographic indicators.
        """
        
        story.append(Paragraph(methodology_text, styles['BodyText']))
        story.append(Spacer(1, 0.2*inch))
        
        # === DISCLAIMERS ===
        story.append(Paragraph("IMPORTANT DISCLAIMERS", styles['SectionTitle']))
        
        disclaimer_text = """
        This report is generated by an automated financial analysis system for informational and educational purposes only. 
        The information should not be construed as investment advice, recommendation, or solicitation to buy or sell securities.
        
        All data is sourced from publicly available information and processed using quantitative algorithms. While every effort 
        is made to ensure accuracy, the completeness and timeliness of information cannot be guaranteed. Past performance 
        does not guarantee future results.
        
        Investment in securities involves substantial risk of loss. Investors should conduct their own due diligence and 
        consult with qualified financial advisors before making investment decisions. This analysis does not consider 
        individual investor circumstances, risk tolerance, or investment objectives.
        
        The automated nature of this analysis means it may not capture all relevant market factors, news events, or 
        qualitative considerations that could materially affect investment outcomes. Users should supplement this analysis 
        with additional research and professional guidance.
        
        Neither the system developers nor the data providers assume any liability for investment decisions made based on 
        this report. All investments carry risk, and investors may lose principal.
        """
        
        story.append(Paragraph(disclaimer_text, styles['Disclaimer']))
        
        # Build the PDF
        doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
        
        logging.info(f"PDF report generated successfully: {pdf_filename}")
        return pdf_filename
        
    except Exception as e:
        logging.error(f"Critical error in PDF generation: {str(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise Exception(f"PDF generation failed: {str(e)}")

# Utility function for backward compatibility
def create_stock_chart(stock_data, output_path):
    """Legacy function for backward compatibility"""
    try:
        price_targets = {}  # Empty price targets for legacy compatibility
        create_comprehensive_price_chart(stock_data, price_targets, output_path)
    except Exception as e:
        logging.error(f"Legacy chart creation failed: {e}")

# Export the main function for external use
__all__ = ['generate_investment_report', 'create_stock_chart']
