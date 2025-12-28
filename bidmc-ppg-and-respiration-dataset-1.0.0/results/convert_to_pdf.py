"""
Convert Markdown PPT content to a properly formatted PDF
"""

import markdown
from markdown.extensions.tables import TableExtension
import os

# Paths
md_file = r"d:\Downloads\bidmc-ppg-and-respiration-dataset-1.0.0-20251011T094008Z-1-001\bidmc-ppg-and-respiration-dataset-1.0.0\results\PPT_CONTENT_10_SLIDES.md"
html_file = r"d:\Downloads\bidmc-ppg-and-respiration-dataset-1.0.0-20251011T094008Z-1-001\bidmc-ppg-and-respiration-dataset-1.0.0\results\PPT_CONTENT_10_SLIDES.html"

# Read markdown content
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
html_body = md.convert(md_content)

# Create full HTML with professional styling for PDF
html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Respiratory Abnormality Detection - First Evaluation Presentation</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0;
            padding: 20px;
            background: #fff;
        }}
        
        h1 {{
            color: #1a5276;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
            font-size: 24px;
            page-break-before: always;
        }}
        
        h1:first-of-type {{
            page-break-before: avoid;
        }}
        
        h2 {{
            color: #2874a6;
            margin-top: 25px;
            font-size: 18px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        
        h3 {{
            color: #1a5276;
            font-size: 16px;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 11px;
            page-break-inside: avoid;
        }}
        
        th {{
            background-color: #3498db;
            color: white;
            padding: 10px 8px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #2980b9;
        }}
        
        td {{
            padding: 8px;
            border: 1px solid #ddd;
            vertical-align: top;
        }}
        
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        tr:hover {{
            background-color: #e8f4f8;
        }}
        
        blockquote {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 15px 0;
            font-style: italic;
            font-weight: 500;
        }}
        
        blockquote p {{
            margin: 0;
        }}
        
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 11px;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        
        pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        
        strong {{
            color: #1a5276;
        }}
        
        em {{
            color: #555;
        }}
        
        hr {{
            border: none;
            height: 2px;
            background: linear-gradient(to right, #3498db, #9b59b6);
            margin: 30px 0;
        }}
        
        ul, ol {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        /* Emoji/Checkmark styling */
        .check {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        /* Page header */
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1a5276 0%, #3498db 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: white;
            border: none;
            margin: 0;
            padding: 0;
            page-break-before: avoid;
        }}
        
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        /* Print specific */
        @media print {{
            body {{
                font-size: 11pt;
            }}
            
            h1 {{
                page-break-before: always;
                font-size: 18pt;
            }}
            
            h1:first-of-type {{
                page-break-before: avoid;
            }}
            
            table {{
                font-size: 9pt;
            }}
            
            pre {{
                font-size: 8pt;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RESPIRATORY ABNORMALITY DETECTION</h1>
        <p>First Evaluation Presentation | December 2025</p>
    </div>
    
    {html_body}
    
    <div class="footer">
        <p><strong>First Evaluation: Data Acquisition → Feature Selection</strong></p>
        <p>Respiratory Abnormality Classification using PPG Signals</p>
    </div>
</body>
</html>
'''

# Save HTML file
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"✅ HTML file created: {html_file}")
print()
print("=" * 60)
print("TO CREATE PDF:")
print("=" * 60)
print()
print("Option 1: Open in Browser and Print to PDF")
print(f"   1. Open the HTML file in your browser")
print(f"   2. Press Ctrl+P (Print)")
print(f"   3. Select 'Save as PDF' as destination")
print(f"   4. Save as 'PPT_CONTENT_10_SLIDES.pdf'")
print()
print("Option 2: Use Microsoft Edge")
print(f"   1. Open HTML in Microsoft Edge")
print(f"   2. Press Ctrl+P → Save as PDF")
print()
print(f"HTML File Location: {html_file}")
