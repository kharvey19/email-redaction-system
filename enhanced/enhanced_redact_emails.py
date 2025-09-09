#!/usr/bin/env python3

import re
import sys
import argparse
import json
import os
import time
from datetime import datetime
from collections import Counter, defaultdict
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor

# Core libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

# NLP Libraries
from textblob import TextBlob
from fuzzywuzzy import fuzz, process
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def escape_html(text):
    """HTML escaping utility function."""
    escape_map = {'<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#x27;'}
    return ''.join(escape_map.get(c, c) for c in text)


def get_sentiment_color_and_text(polarity):
    """Get sentiment text and color based on polarity."""
    if polarity > 0.1:
        return "Positive", '#28A745'  # Green
    elif polarity < -0.1:
        return "Negative", '#DC3545'  # Red
    else:
        return "Neutral", '#6C757D'   # Gray


class IntelligentRedactor:
    """Enhanced redaction system with NLP capabilities."""
    
    def __init__(self, student_names=None, confidence_threshold=0.85):
        self.student_names = student_names or []
        self.confidence_threshold = confidence_threshold
        self.redaction_stats = defaultdict(int)
        self.email_analytics = {
            'sentiments': [],
            'categories': [],
            'urgency_levels': [],
            'entities_found': [],
            'word_frequencies': Counter()
        }
        self.timing_data = {
            'individual_redactions': [],
            'total_redaction_time': 0,
            'pdf_generation_time': 0,
            'chart_generation_time': 0,
            'total_processing_time': 0,
            'document_creation_time': 0
        }
        
        
        # Pre-compiled patterns
        self._student_email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@student\.d123\.org\b', re.IGNORECASE)
        self.redaction_style = '[REDACTED]'


    def fuzzy_match_names(self, text, threshold=85):
        """Use fuzzy matching to find name variations and misspellings."""
        if not self.student_names:
            return []
        
        words = re.findall(r'\b[A-Za-z]+(?:\s+[A-Za-z]+)*\b', text)
        matches = []
        
        for word_sequence in words:
            for name in self.student_names:
                ratio = fuzz.ratio(word_sequence.lower(), name.lower())
                if ratio >= threshold:
                    matches.append({
                        'text': word_sequence,
                        'matched_name': name,
                        'confidence': ratio / 100.0,
                        'type': 'fuzzy_name'
                    })
        
        return matches

    def analyze_sentiment(self, text):
        """Analyze sentiment and extract emotional indicators."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine urgency based on keywords and sentiment
        urgency_keywords = {
            'high': ['urgent', 'emergency', 'immediate', 'asap', 'critical', 'severe'],
            'medium': ['soon', 'important', 'needed', 'required'],
            'low': ['whenever', 'eventually', 'when possible']
        }
        
        text_lower = text.lower()
        urgency = 'medium'  # default
        
        for level, keywords in urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                urgency = level
                break
        
        # Adjust urgency based on sentiment
        if polarity < -0.3:  # Very negative
            urgency = 'high' if urgency != 'low' else 'medium'
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'urgency': urgency,
            'description': self._sentiment_description(polarity, subjectivity)
        }

    def _sentiment_description(self, polarity, subjectivity):
        """Convert numerical sentiment to descriptive text."""
        if polarity > 0.1:
            tone = "positive"
        elif polarity < -0.1:
            tone = "negative" 
        else:
            tone = "neutral"
        
        if subjectivity > 0.5:
            objectivity = "subjective"
        else:
            objectivity = "objective"
        
        return f"{tone} and {objectivity}"

    def categorize_email(self, text, subject=""):
        """Categorize email by department/topic using keyword analysis."""
        categories = {
            'Academic': ['grade', 'assignment', 'homework', 'test', 'exam', 'class', 'course', 'teacher'],
            'Health': ['allergy', 'medical', 'nurse', 'sick', 'health', 'medication', 'injury'],
            'Transportation': ['bus', 'transportation', 'pickup', 'drop-off', 'route'],
            'Behavioral': ['bullying', 'behavior', 'discipline', 'incident', 'counselor'],
            'Administrative': ['enrollment', 'registration', 'form', 'document', 'office'],
            'Extracurricular': ['club', 'sport', 'activity', 'event', 'team', 'practice'],
            'Technology': ['computer', 'internet', 'device', 'software', 'login'],
            'Library': ['book', 'library', 'reading', 'research']
        }
        
        text_combined = (subject + " " + text).lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_combined)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'General'

    def redact(self, text):
        """Apply standard redaction for names and emails only."""
        start_time = time.time()
        redacted_text = text
        
        # 1. Student name redaction 
        if self.student_names:
            escaped_names = [re.escape(name) for name in self.student_names]
            pattern = re.compile(r'\b(?:' + '|'.join(escaped_names) + r')\b', re.IGNORECASE)
            
            def replace_name(match):
                self.redaction_stats['names'] += 1
                return self.redaction_style
            
            redacted_text = pattern.sub(replace_name, redacted_text)
            
            # Fuzzy name matching
            fuzzy_matches = self.fuzzy_match_names(redacted_text)
            for match in fuzzy_matches:
                if match['confidence'] > self.confidence_threshold:
                    redacted_text = redacted_text.replace(match['text'], self.redaction_style)
                    self.redaction_stats['fuzzy_names'] += 1
        
        # 2. Email redaction
        def replace_email(match):
            self.redaction_stats['emails'] += 1
            return self.redaction_style
        
        redacted_text = self._student_email_pattern.sub(replace_email, redacted_text)
        
        end_time = time.time()
        redaction_time = end_time - start_time
        self.timing_data['individual_redactions'].append(redaction_time)
        
        return redacted_text, []

    def generate_analytics_summary(self):
        """Generate a comprehensive analytics summary."""
        total_redactions = sum(self.redaction_stats.values())
        
        summary = {
            'total_redactions': total_redactions,
            'redaction_breakdown': dict(self.redaction_stats),
            'email_analytics': self.email_analytics,
            'timing_data': self.timing_data,
            'unique_students': len(self.student_names) if self.student_names else 0,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return summary


class EnhancedPDFGenerator:
    """Enhanced PDF generator with analytics and visualizations."""
    
    def __init__(self):
        self.styles = self._create_enhanced_styles()
    
    def _create_enhanced_styles(self):
        """Create simple PDF styles without colors."""
        base_styles = getSampleStyleSheet()
        
        styles = {
            'title': ParagraphStyle(
                'SimpleTitle',
                parent=base_styles['Title'],
                fontName='Helvetica-Bold',
                fontSize=18,
                spaceAfter=30,
                alignment=1
            ),
            'subtitle': ParagraphStyle(
                'Subtitle',
                parent=base_styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=14,
                spaceAfter=20
            ),
            'email_header': ParagraphStyle(
                'EmailHeader',
                parent=base_styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                spaceAfter=4
            ),
            'email_body': ParagraphStyle(
                'EmailBody',
                parent=base_styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                spaceAfter=6,
                leftIndent=0
            ),
            'analytics': ParagraphStyle(
                'Analytics',
                parent=base_styles['Normal'],
                fontName='Helvetica',
                fontSize=9
            ),
            'redacted': ParagraphStyle(
                'Redacted',
                parent=base_styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                spaceAfter=6,
                leftIndent=0
            ),
            'separator': ParagraphStyle(
                'Separator',
                parent=base_styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                spaceAfter=12,
                spaceBefore=12,
                alignment=1
            )
        }
        
        return styles

    def _create_horizontal_bar_chart(self, data_dict, title, output_path):
        """Create a horizontal bar chart for categories using matplotlib."""
        
        labels = list(data_dict.keys())
        values = list(data_dict.values())
        
        if not values or sum(values) == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ripped_colors = ['#FF6B35', '#F7931E', '#FFD23F', '#5D8AA8', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        bar_colors = ripped_colors[:len(labels)]
        bars = ax.barh(labels, values, color=bar_colors, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                    ha='left', va='center', fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Emails')
        ax.set_ylabel('Category')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path

    def _create_combined_pie_chart(self, sentiment_data, urgency_data, output_path):
        """Create combined subplot with sentiment and urgency pie charts using matplotlib."""
        
        # Check if we have data
        if (not sentiment_data or sum(sentiment_data.values()) == 0) and \
            (not urgency_data or sum(urgency_data.values()) == 0):
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Define specific colors for sentiment and urgency  
        sentiment_color_map = {
            'Positive': '#28A745',  # Green
            'Negative': '#DC3545',  # Red  
            'Neutral': '#D3D3D3'    # Light Gray
        }
        
        urgency_color_map = {
            'High': '#9B59B6',     # Light Purple
            'Medium': '#F1C40F',   # Light Yellow
            'Low': '#3498DB'       # Light Blue
        }
        
        # Sentiment pie chart
        if sentiment_data and sum(sentiment_data.values()) > 0:
            sentiment_colors = [sentiment_color_map.get(label, '#6C757D') for label in sentiment_data.keys()]
            ax1.pie(sentiment_data.values(), labels=sentiment_data.keys(), 
                    autopct='%1.1f%%', startangle=90, colors=sentiment_colors)
            ax1.set_title('Sentiment Analysis', fontweight='bold')
        
        # Urgency pie chart
        if urgency_data and sum(urgency_data.values()) > 0:
            urgency_colors = [urgency_color_map.get(label, '#6C757D') for label in urgency_data.keys()]
                
            ax2.pie(urgency_data.values(), labels=urgency_data.keys(), 
                    autopct='%1.1f%%', startangle=90, colors=urgency_colors)
            ax2.set_title('Urgency Levels', fontweight='bold')
        
        plt.suptitle('Sentiment & Urgency Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


    def create_enhanced_pdf(self, emails_data, output_file, pie_chart_paths=None):
        """Create a simple PDF with pie charts and priority-sorted emails."""
        doc = SimpleDocTemplate(
            output_file, 
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )
        
        story = []
        
        # Simple title
        story.append(Paragraph("Redacted Email Report", self.styles['title']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['analytics']))
        story.append(Spacer(1, 30))
        
        # Add charts if available
        if pie_chart_paths:
            story.append(Paragraph("Email Analytics Overview", self.styles['subtitle']))
            
            for chart_path in pie_chart_paths:
                chart_img = Image(chart_path, width=6.5*inch, height=3.25*inch)
                story.append(chart_img)
                story.append(Spacer(1, 15))

        story.append(PageBreak())

        
        # Sort emails by priority (high -> medium -> low)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_emails = sorted(emails_data, key=lambda x: priority_order.get(x.get('sentiment', {}).get('urgency', 'medium'), 1))
        
        # Redacted emails section
        story.append(Paragraph(f"Redacted Emails (Sorted by Priority)", self.styles['subtitle']))
        story.append(Spacer(1, 20))
        
        # Process emails in parallel for PDF generation
        total_emails = len(sorted_emails)
        
        # Prepare arguments for parallel processing
        email_args = [(email_data, i, total_emails) for i, email_data in enumerate(sorted_emails)]
        
        # Use ProcessPoolExecutor for parallel PDF element generation
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), len(sorted_emails), 8)) as executor:
            processed_results = list(executor.map(format_email_for_pdf, email_args))
        
        # Combine results into PDF story in order
        for result in processed_results:
            # Create colored sentiment tag
            sentiment_color = HexColor(result['sentiment_color'])
            sentiment_tag_style = ParagraphStyle(
                'SentimentTag',
                parent=self.styles['analytics'],
                textColor=sentiment_color,
                fontName='Helvetica-Bold'
            )
            
            # Add sentiment tags
            story.append(Paragraph(result['tags'], sentiment_tag_style))
            story.append(Spacer(1, 8))
            
            # Process content lines
            for line_type, line_content in result['processed_lines']:
                if line_type == 'spacer':
                    story.append(Spacer(1, line_content))
                elif line_type == 'header':
                    story.append(Paragraph(f'<b>{line_content}</b>', self.styles['email_header']))
                elif line_type == 'redacted':
                    story.append(Paragraph(line_content, self.styles['redacted']))
                elif line_type == 'body':
                    story.append(Paragraph(line_content, self.styles['email_body']))
            
            # Add separator between emails (but not after last email)
            if not result['is_last_email']:
                story.append(Spacer(1, 12))
                story.append(Paragraph('_' * 80, self.styles['separator']))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        return output_file


def format_email_for_pdf(args):
    """Format a processed email into PDF-ready elements with styling and layout."""
    email_data, email_index, total_emails = args
    
    # Email metadata with sentiment information
    category = email_data.get('category', 'General')
    sentiment = email_data.get('sentiment', {})
    urgency = sentiment.get('urgency', 'medium')
    polarity = sentiment.get('polarity', 0)
    
    # Determine sentiment color and text
    sentiment_text, sentiment_color_hex = get_sentiment_color_and_text(polarity)
    
    # Enhanced tags with sentiment
    tags = f"Category: {category} | Priority: {urgency.title()} | Sentiment: {sentiment_text}"
    
    # Email content processing
    lines = email_data['content'].strip().split('\n')
    header_patterns = ('From:', 'To:', 'CC:', 'Subject:', 'Date:')
    
    processed_lines = []
    for line in lines:
        if not line.strip():
            processed_lines.append(('spacer', 6))
        elif any(line.startswith(pattern) for pattern in header_patterns):
            escaped_line = escape_html(line)
            processed_lines.append(('header', escaped_line))
        else:
            escaped_line = escape_html(line)
            if '[REDACTED]' in escaped_line:
                processed_lines.append(('redacted', escaped_line))
            else:
                processed_lines.append(('body', escaped_line))
    
    return {
        'tags': tags,
        'sentiment_color': sentiment_color_hex,
        'processed_lines': processed_lines,
        'email_index': email_index,
        'is_last_email': email_index == total_emails - 1
    }


def process_email_content(args):
    """Process and analyze a single email's content (redaction, sentiment, categorization)."""
    email_content, student_names = args
    
    # Create a temporary redactor for this process
    redactor = IntelligentRedactor(student_names)
    
    start_time = time.time()
    
    # Perform redaction
    redacted_content, redactions_count = redactor.redact(email_content)
    
    # Analyze sentiment and categorize
    sentiment_data = redactor.analyze_sentiment(email_content)
    category = redactor.categorize_email(email_content)
    
    processing_time = time.time() - start_time
    
    return {
        'content': redacted_content,
        'redactions': dict(redactor.redaction_stats),  # Return the stats dictionary
        'sentiment': sentiment_data,
        'category': category,
        'processing_time': processing_time
    }

def main_email_processor(input_file, output_file, student_names=None, use_parallel=False, max_workers=None):
    """Main email processing pipeline with redaction, NLP analysis, and PDF generation."""
    overall_start_time = time.time()
    
    # Initialize the intelligent redactor
    redactor = IntelligentRedactor(student_names)
    pdf_generator = EnhancedPDFGenerator()
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into email blocks
    email_blocks = [block.strip() for block in content.split('\n---\n') if block.strip()]
    
    processed_emails = []
    redaction_start_time = time.time()
    
    if use_parallel and len(email_blocks) > 1:
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(email_blocks), 8)  # Cap at 8 to avoid overhead
        
        print(f"Processing {len(email_blocks)} emails using {max_workers} parallel workers...")
        
        # Prepare arguments for parallel processing
        email_args = [(email_block, student_names) for email_block in email_blocks]
        
        # Use ProcessPoolExecutor for CPU-intensive tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_email_content, email_args))
        
        # Combine results and update main redactor stats
        for i, result in enumerate(results):
            # Extract subject for categorization
            subject_match = re.search(r'Subject:\s*(.+)', email_blocks[i])
            subject = subject_match.group(1) if subject_match else ""
            
            processed_emails.append({
                'content': result['content'],
                'sentiment': result['sentiment'],
                'category': result['category'],
                'redactions': result['redactions'],
                'original_length': len(email_blocks[i]),
                'redacted_length': len(result['content'])
            })
            
            # Update analytics data
            redactor.email_analytics['sentiments'].append(result['sentiment'])
            redactor.email_analytics['categories'].append(result['category'])
            redactor.email_analytics['urgency_levels'].append(result['sentiment']['urgency'])
            
            # Update timing data
            redactor.timing_data['individual_redactions'].append(result['processing_time'])
            
            # Update redaction stats (accumulate from parallel workers)  
            for redaction_type_key, count in result['redactions'].items():
                redactor.redaction_stats[redaction_type_key] += count
    else:
        # Sequential processing (original method)
        print(f"Processing {len(email_blocks)} emails sequentially...")
        
        for i, email_block in enumerate(email_blocks):
            
            # Extract subject for better categorization
            subject_match = re.search(r'Subject:\s*(.+)', email_block)
            subject = subject_match.group(1) if subject_match else ""
            
            # Apply intelligent redaction (simplified)
            redacted_content, redactions = redactor.redact(email_block)
            
            # Analyze sentiment on original content (before redaction for accuracy)
            sentiment_analysis = redactor.analyze_sentiment(email_block)
            
            # Categorize email
            category = redactor.categorize_email(email_block, subject)
            
            # Update analytics
            redactor.email_analytics['sentiments'].append(sentiment_analysis)
            redactor.email_analytics['categories'].append(category)
            redactor.email_analytics['urgency_levels'].append(sentiment_analysis['urgency'])
            
            processed_emails.append({
                'content': redacted_content,
                'category': category,
                'sentiment': sentiment_analysis,
                'redactions': redactions,
                'original_length': len(email_block),
                'redacted_length': len(redacted_content)
            })
    
    # Record total redaction time
    redaction_end_time = time.time()
    redactor.timing_data['total_redaction_time'] = redaction_end_time - redaction_start_time
    
    # Generate analytics summary
    analytics_summary = redactor.generate_analytics_summary()
    
    # Categories data
    categories = {}
    for email in processed_emails:
        cat = email.get('category', 'General')
        categories[cat] = categories.get(cat, 0) + 1
    
    # Sentiment data
    sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for email in processed_emails:
        sentiment = email.get('sentiment', {})
        polarity = sentiment.get('polarity', 0)
        if polarity > 0.1:
            sentiments['Positive'] += 1
        elif polarity < -0.1:
            sentiments['Negative'] += 1
        else:
            sentiments['Neutral'] += 1
    
    # Urgency data
    urgency_levels = {'High': 0, 'Medium': 0, 'Low': 0}
    for email in processed_emails:
        urgency = email.get('sentiment', {}).get('urgency', 'medium')
        urgency_levels[urgency.title()] += 1
    
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Create modern charts with better naming
    chart_start_time = time.time()
    pie_chart_paths = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Horizontal bar chart for categories
    if categories:
        chart_path = pdf_generator._create_horizontal_bar_chart(
            categories, 
            'Email Categories', 
            os.path.join('figures', f"email_categories_{timestamp}.png")
        )
        if chart_path:
            pie_chart_paths.append(chart_path)
    
    # Combined subplot for sentiment and urgency
    if any(sentiments.values()) or any(urgency_levels.values()):
        chart_path = pdf_generator._create_combined_pie_chart(
            sentiments, 
            urgency_levels, 
            os.path.join('figures', f"sentiment_urgency_{timestamp}.png")
        )
        if chart_path:
            pie_chart_paths.append(chart_path)
    
    chart_end_time = time.time()
    redactor.timing_data['chart_generation_time'] = chart_end_time - chart_start_time
    
    # Generate PDF with pie charts
    pdf_start_time = time.time()
    pdf_generator.create_enhanced_pdf(
        processed_emails, 
        output_file,
        pie_chart_paths,
    )
    pdf_end_time = time.time()
    redactor.timing_data['pdf_generation_time'] = pdf_end_time - pdf_start_time
    
    # Calculate total document creation time (charts + PDF)
    redactor.timing_data['document_creation_time'] = (chart_end_time - chart_start_time) + (pdf_end_time - pdf_start_time)
    
    # Calculate total processing time
    overall_end_time = time.time()
    redactor.timing_data['total_processing_time'] = overall_end_time - overall_start_time
    
    # Update analytics summary with final timing data
    analytics_summary = redactor.generate_analytics_summary()
    
    # Save analytics as JSON for further analysis
    analytics_file = output_file.replace('.pdf', '_analytics.json')
    with open(analytics_file, 'w', encoding='utf-8') as f:
        json.dump(analytics_summary, f, indent=2, default=str)
    
    print("Document generated!")
    
    return output_file, analytics_file


def main():
    """Enhanced main function with additional options."""
    parser = argparse.ArgumentParser(
        description='Enhanced email redaction with NLP analysis and intelligent privacy protection'
    )
    parser.add_argument('input_file', help='Input text file containing email content')
    parser.add_argument('output_file', help='Output PDF file for redacted content')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='Enable parallel processing (default: sequential)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect)')
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.85,
        help='Confidence threshold for fuzzy matching (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Default student names
    student_names = [
        "Clara Benson", "Hunter Schaffer", "Grace Johnson", "Kyle Simmons",
        "Maya Patel", "Elena Gomez", "Jamal White", "Ryan Lee",
        "Isabella Torres", "Omar Nasser", "Ava Martinez", "Noah Kim"
    ]
    
    # Process emails with NLP
    try:
        _ = main_email_processor(
            args.input_file, 
            args.output_file, 
            student_names,
            use_parallel=args.parallel,
            max_workers=args.workers
        )
        
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
