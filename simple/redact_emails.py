#!/usr/bin/env python3

import re
import sys
import argparse
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from functools import lru_cache


# Pre-compiled regex patterns
_student_email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@student\.d123\.org\b', re.IGNORECASE)
_html_escape_map = str.maketrans({'<': '&lt;', '>': '&gt;', '&': '&amp;'})

def redact_student_names(text, student_names):
    """
    Redact the specified student names (case-insensitive) from the text.
    Uses a single compiled regex pattern for all names for better performance.
    
    Args:
        text (str): The input text to redact
        student_names (list): List of student names to redact
        
    Returns:
        str: Text with student names redacted
    """
    if not student_names:
        return text
    
    # Create a single regex pattern for all names using alternation
    escaped_names = [re.escape(name) for name in student_names]
    pattern = re.compile(r'\b(?:' + '|'.join(escaped_names) + r')\b', re.IGNORECASE)
    
    return pattern.sub('[REDACTED]', text)


def redact_student_emails(text):
    """
    Redact student email addresses that match the pattern @student.d123.org
    Uses pre-compiled regex pattern for better performance.
    
    Args:
        text (str): The input text to redact
        
    Returns:
        str: Text with student emails redacted
    """
    return _student_email_pattern.sub('[REDACTED]', text)


@lru_cache(maxsize=1)
def _get_pdf_styles():
    """Cache PDF styles to avoid recreating them for each call."""
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=16,
        spaceAfter=24,
        alignment=1
    )
    
    email_style = ParagraphStyle(
        'EmailStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        spaceAfter=6,
        leftIndent=0
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        spaceAfter=6,
        leftIndent=0
    )
    
    separator_style = ParagraphStyle(
        'SeparatorStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        spaceAfter=12,
        spaceBefore=12,
        alignment=1
    )
    
    return title_style, email_style, header_style, separator_style

def _escape_html(text):
    """Fast HTML escaping using str.translate."""
    return text.translate(_html_escape_map)

def _process_email_block(email_text, header_style, email_style):
    """
    Process a single email block and return PDF elements.
    
    Args:
        email_text (str): Text content of a single email
        header_style: PDF style for headers
        email_style: PDF style for regular content
        
    Returns:
        list: List of PDF elements for this email
    """
    elements = []
    lines = email_text.strip().split('\n')
    header_patterns = ('From:', 'To:', 'CC:', 'Subject:', 'Date:')
    
    for line in lines:
        if not line.strip():
            elements.append(Spacer(1, 6))
        elif any(line.startswith(pattern) for pattern in header_patterns):
            escaped_line = _escape_html(line)
            elements.append(Paragraph(f'<b>{escaped_line}</b>', header_style))
        else:
            escaped_line = _escape_html(line)
            elements.append(Paragraph(escaped_line, email_style))
    
    return elements

def create_pdf_from_text(text_content, output_file):
    """
    Create a PDF file from the redacted text content.
    Improved version using email block splitting for better structure and maintainability.
    
    Args:
        text_content (str): The redacted text content
        output_file (str): Path to the output PDF file
    """
    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    title_style, email_style, header_style, separator_style = _get_pdf_styles()
    
    # Build the story
    story = []
    story.append(Paragraph("Redacted Emails", title_style))
    story.append(Spacer(1, 12))
    
    email_blocks = [block.strip() for block in text_content.split('\n---\n') if block.strip()]
    
    for i, email_block in enumerate(email_blocks):
        # Process each email block and add to the story
        email_elements = _process_email_block(email_block, header_style, email_style)
        story.extend(email_elements)
        
        # Add separator between emails, but not after the last one
        if i < len(email_blocks) - 1:
            story.append(Spacer(1, 12))
            story.append(Paragraph('_' * 80, separator_style))
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)


def redact_email_content(input_file, output_file, student_names):
    """
    Process the email content file and create a redacted PDF version.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to the output PDF file
        student_names (list): List of student names to redact
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply redactions in sequence
    redacted_content = redact_student_names(content, student_names)
    redacted_content = redact_student_emails(redacted_content)
    
    # Create PDF
    create_pdf_from_text(redacted_content, output_file)



def main():
    """
    Main function to handle command line arguments and run the redaction process.
    """
    parser = argparse.ArgumentParser(
        description='Redact student names and emails from email content file'
    )
    parser.add_argument(
        'input_file', 
        help='Input text file containing email content'
    )
    parser.add_argument(
        'output_file', 
        help='Output PDF file for redacted content'
    )

    student_names = [
        "Clara Benson", "Hunter Schaffer", "Grace Johnson", "Kyle Simmons",
        "Maya Patel", "Elena Gomez", "Jamal White", "Ryan Lee",
        "Isabella Torres", "Omar Nasser", "Ava Martinez", "Noah Kim"
    ]
    
    args = parser.parse_args()
    
    redact_email_content(args.input_file, args.output_file, student_names)



if __name__ == "__main__":
    main()
