
![Email Redaction System](email.webp)
# Email Redaction System

A privacy-focused email processing system that automatically redacts sensitive student information from email communications. The system offers both basic sequential processing and advanced parallel processing with NLP-powered sentiment analysis and categorization.

## File Structure

**Main Directory:**
- `deduplicated_messages.txt` - Input email data
- `email.webp` - README header image
- `requirements.txt` - Dependencies
- `simple/` - Basic redaction
  - `redact_emails.py` - Simple redaction script
  - `redacted_emails.pdf` - Sample output
- `enhanced/` - Advanced redaction with analytics
  - `enhanced_redact_emails.py` - Advanced redaction script
  - `email_analysis.ipynb` - Jupyter notebook for analysis
  - `performance_summary.md` - Performance summary
  - `figures/` - Generated visualization charts
  - `output.pdf` - Sample enhanced output
  - `output_analytics.json` - Analytics data

## Simple Version

Basic sequential email redaction that removes student names and emails.

**To run:**
```bash
cd simple
python redact_emails.py ../deduplicated_messages.txt output.pdf
```

## Enhanced Version

Adds NLP sentiment analysis, email categorization, parallel processing, and interactive charts.

**To run:**
```bash
cd enhanced  
python enhanced_redact_emails.py ../deduplicated_messages.txt output.pdf
```

## How the Logic Works

**Simple:** The basic version uses sequential processing with Python's `re` module for exact string matching of student names and email patterns. It generates output using ReportLab for straightforward PDF creation without additional analysis features.

**Enhanced:** The advanced version leverages parallel processing through `ProcessPoolExecutor` from the multiprocessing library to handle multiple emails simultaneously. It utilizes fuzzy name matching using the `fuzzywuzzy` library with Levenshtein distance algorithms to catch misspelled names. Sentiment analysis is performed via `TextBlob` for polarity and subjectivity scoring, while category detection uses keyword matching against predefined dictionaries. The system generates interactive charts using `Plotly` and `matplotlib`, and includes comprehensive performance timing and analytics collection throughout the processing pipeline.

Note: For enhanced logic, classifications for urgency and sentiment analysis might be slighly incorrect. This implementaiton was left basic since it is an additional feature, would be interested in further exploring these metrics. 

## Assumptions Made
- Student emails follow @student.d123.org pattern
- Email blocks separated by `---`
- Names provided are accurate

## Edge Cases Considered
- Misspelled names (fuzzy matching)
- Various email formats
- Empty/malformed emails

## Insights
- Parallel processing 3-5x faster than sequential
- Users filter out which emails they want to look at by looking at urgency and sentiment
- Could be further optimized with GPU parallel processing

## Possible Extensions
- GPU acceleration for massive datasets
- Real-time processing pipelines
- ML-based entity recognition