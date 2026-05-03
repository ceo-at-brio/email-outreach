import pandas as pd
import re
import json
import socket
import requests
from datetime import datetime
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import dns.resolver

    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util

    ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, downloaded once
    ST_AVAILABLE = True
    print('Semantic model loaded: all-MiniLM-L6-v2')
except ImportError:
    ST_AVAILABLE = False
    ST_MODEL = None
    print('sentence-transformers not installed. Run: pip install sentence-transformers')

try:
    from keybert import KeyBERT

    KB_MODEL = KeyBERT(model=ST_MODEL) if ST_MODEL else None
    KEYBERT_AVAILABLE = KB_MODEL is not None
    if KEYBERT_AVAILABLE:
        print('KeyBERT loaded for bold phrase extraction')
except ImportError:
    KB_MODEL = None
    KEYBERT_AVAILABLE = False
    print('keybert not installed. Run: pip install keybert')

# File name
file_path = 'Email-Script - 23_3.csv'


def add_bold_formatting(text, top_n=2):
    """
    Extracts the top N most important keyphrases from text using KeyBERT
    and wraps them in <b> tags.
    Falls back to bold_important_words() if KeyBERT is not available.
    """
    if not text or not text.strip():
        return text

    if KEYBERT_AVAILABLE:
        try:
            keywords = KB_MODEL.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=top_n
            )
            if keywords:
                result = text
                # Sort keywords by length descending to help avoid messy replacements
                for kw, score in sorted(keywords, key=lambda x: -len(x[0])):
                    # Avoid bolding if already inside a <b> tag
                    pattern = re.compile(rf'(?<!<b>){re.escape(kw)}(?!</b>)', re.IGNORECASE)
                    result = pattern.sub(lambda m: f'<b>{m.group(0)}</b>', result, count=1)
                return result
        except Exception:
            pass

    return bold_important_words(text, target_words=top_n * 2)


def bold_important_words(text, target_words=7):
    """
    Bolds the most important words in text using a pure-Python heuristic.
    """
    if not text or not text.strip():
        return text

    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'i', 'we', 'you', 'he',
        'she', 'it', 'they', 'my', 'our', 'your', 'their', 'this', 'that',
        'these', 'those', 'after', 'before', 'during', 'through', 'about',
        'how', 'what', 'when', 'where', 'who', 'which', 'quick', 'just',
        'note', 'earlier', 'follow', 'up', 'its', 'also', 'as', 'so', 'if',
        'not', 'no', 'into', 'then', 'than', 'very', 'more', 'most', 'some',
    }

    tokens = list(re.finditer(r'\b[a-zA-Z][\w\'-]*\b', text))
    scored = []
    for i, match in enumerate(tokens):
        word = match.group(0)
        clean = word.lower().rstrip("'s").rstrip("'")
        if clean in STOP_WORDS or len(clean) < 3:
            continue
        score = len(clean)
        if word[0].isupper() and i > 0:
            score += 8
        scored.append((match.start(), match.end(), word, score))

    top = sorted(scored, key=lambda x: -x[3])[:target_words]

    result = text
    for start, end, word, _ in sorted(top, key=lambda x: -x[0]):
        # Prevent nesting
        if '<b>' in result[max(0, start - 4):end + 4]: continue
        result = result[:start] + f'<b>{result[start:end]}</b>' + result[end:]

    return result


def bold_last_paragraph(text):
    """
    Bolds 'free tech proposal' and the 3-4 words that follow it.
    """
    if not text or not text.strip():
        return text

    # Match 'free tech proposal' (case-insensitive) + 3-4 words after it
    pattern = re.compile(
        r'(free tech proposal(?:\s+\w+){3,4})',
        re.IGNORECASE
    )
    return pattern.sub(lambda m: f'<b>{m.group(0)}</b>', text, count=1)


# Patterns that identify a CTA (call-to-action) sentence in Last Paragraph
_CTA_PATTERNS = re.compile(
    r'open to|worth a|happy to|would you be|let me know|feel free|'
    r'brief intro|quick intro|quick call|brief call|15.min|short call|'
    r'free tech proposal|free proposal',
    re.IGNORECASE
)


def reorder_last_paragraph(text):
    """
    Guarantees that any CTA sentence (e.g. 'Open to a brief intro?') is always
    the LAST sentence in Last Paragraph.

    If the CTA sentence appears first (before the 'We can help...' line), the
    two are swapped so the CTA always closes the paragraph.
    """
    if not text or not text.strip():
        return text

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2:
        return text  # nothing to reorder

    def _is_cta(s):
        return bool(_CTA_PATTERNS.search(s))

    cta_parts     = [p for p in parts if _is_cta(p)]
    non_cta_parts = [p for p in parts if not _is_cta(p)]

    if not cta_parts or not non_cta_parts:
        return text  # all CTA or no CTA — leave as-is

    # Non-CTA content first, CTA sentences last
    return ' '.join(non_cta_parts + cta_parts)


def compute_email_relevance(row):
    """
    Computes semantic relevance between ALL email content and the prospect's context.
    """
    if not ST_AVAILABLE:
        return {
            'relevance_score': '',
            'relevance_label': 'N/A',
            'relevance_detail': 'Install sentence-transformers to enable this feature'
        }

    # Use the full raw email text from the original CSV columns
    email_cols = ['Introductory mail', '1st Followup', '2nd Followup']
    email_parts = [str(row.get(col, '') or '').strip() for col in email_cols]
    email_text = ' '.join(p for p in email_parts if p)

    # Prospect context columns — try common naming variations
    def get_col(row, *candidates):
        for name in candidates:
            val = row.get(name, None)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ''

    service_text = get_col(row, 'service outsourced', 'Service Outsourced',
                           'Service outsourced', 'services outsourced')
    review_text = get_col(row, 'review text', 'Review Text', 'Review text',
                          'review', 'Review')
    prospect_text = ' '.join(p for p in [service_text, review_text] if p)

    if not email_text:
        return {
            'relevance_score': 0.0,
            'relevance_label': 'N/A',
            'relevance_detail': 'Missing parsed email content'
        }
    if not prospect_text:
        return {
            'relevance_score': 0.0,
            'relevance_label': 'N/A',
            'relevance_detail': 'Missing prospect context (service outsourced / review text columns not found)'
        }

    # Encode both texts into semantic vectors
    email_embedding = ST_MODEL.encode(email_text, convert_to_tensor=True)
    prospect_embedding = ST_MODEL.encode(prospect_text, convert_to_tensor=True)

    # Cosine similarity: 1.0 = identical meaning, 0.0 = unrelated
    score = float(st_util.cos_sim(email_embedding, prospect_embedding)[0][0])
    score = round(score, 3)

    # Label thresholds
    if score >= 0.65:
        label = 'High'
        detail = f'Score {score} — All 3 emails strongly aligned with prospect context'
    elif score >= 0.40:
        label = 'Medium'
        detail = f'Score {score} — Emails moderately relevant to prospect context'
    else:
        label = 'Low'
        detail = f'Score {score} — Emails may not resonate well with this prospect'

    return {
        'relevance_score': score,
        'relevance_label': label,
        'relevance_detail': detail
    }


# Shared headers to mimic a real browser
REQUEST_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def validate_and_read_link(url):
    """
    Validates a URL by checking its HTTP status code.
    """
    empty_result = {
        'link_valid': False,
        'link_status': '',
        'link_final_url': ''
    }

    if not url or not isinstance(url, str) or not url.strip():
        empty_result['link_status'] = 'no link'
        return empty_result

    url = url.strip()

    # Basic URL format check
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        empty_result['link_status'] = 'invalid URL format'
        return empty_result

    def check_status(method='HEAD'):
        fn = requests.head if method == 'HEAD' else requests.get
        return fn(
            url,
            headers=REQUEST_HEADERS,
            timeout=10,
            allow_redirects=True,
            **({'stream': True} if method == 'GET' else {})
        )

    try:
        response = check_status('HEAD')
        if response.status_code == 405:
            response = check_status('GET')

        status_code = response.status_code
        final_url = response.url

        if status_code < 400:
            return {
                'link_valid': True,
                'link_status': f'{status_code} Active',
                'link_final_url': final_url
            }
        if status_code == 403:
            return {
                'link_valid': True,
                'link_status': '403 Active (bot-protected, opens in browser)',
                'link_final_url': final_url
            }

        error_labels = {
            404: '404 Not Found',
            410: '410 Gone',
            500: '500 Server Error',
            502: '502 Bad Gateway',
            503: '503 Service Unavailable',
        }
        label = error_labels.get(status_code, f'{status_code} Error')
        return {
            'link_valid': False,
            'link_status': label,
            'link_final_url': final_url
        }

    except requests.exceptions.SSLError:
        return {**empty_result, 'link_status': 'SSL certificate error'}
    except requests.exceptions.ConnectionError:
        return {**empty_result, 'link_status': 'Connection error (domain unreachable)'}
    except requests.exceptions.Timeout:
        return {**empty_result, 'link_status': 'Request timed out'}
    except Exception as e:
        return {**empty_result, 'link_status': f'Error: {str(e)}'}


def validate_email_address(email):
    """
    Validate an email address using Regex and MX record lookup.
    """
    if not email or not isinstance(email, str):
        return False, 'empty'

    email = email.strip()
    email_regex = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return False, 'invalid format'

    domain = email.split('@')[1]

    if DNS_AVAILABLE:
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            if mx_records:
                return True, 'valid (MX found)'
            else:
                return False, 'no MX record'
        except dns.resolver.NXDOMAIN:
            return False, 'domain does not exist'
        except dns.resolver.NoAnswer:
            return False, 'no MX record'
        except dns.resolver.Timeout:
            return False, 'DNS timeout'
        except Exception:
            try:
                socket.gethostbyname(domain)
                return True, 'valid (A record found, no MX)'
            except socket.gaierror:
                return False, 'domain unreachable'
    else:
        try:
            socket.gethostbyname(domain)
            return True, 'valid (format ok, domain reachable)'
        except socket.gaierror:
            return False, 'domain unreachable'


def validate_emails_field(email_field):
    """
    Handle email fields that may contain multiple comma-separated emails.
    """
    if not email_field or not isinstance(email_field, str):
        return False, 'empty'

    emails = [e.strip() for e in email_field.split(',') if e.strip()]
    statuses = []
    all_valid = True

    for email in emails:
        is_valid, status = validate_email_address(email)
        statuses.append(f'{email}: {status}')
        if not is_valid:
            all_valid = False

    return all_valid, ' | '.join(statuses)


def parse_intro_mail(intro_text):
    """
    Parse introductory mail.
    """
    if pd.isna(intro_text) or not intro_text.strip():
        return {'Name': '', 'Intro Line': '', 'Middle Body': '', 'Last Paragraph': ''}

    text = str(intro_text).strip()
    lines = text.split('\n')

    if len(lines) > 1:
        last_line = lines[-1].strip()
        signature_indicators = [
            len(last_line) < 150 and (',' in last_line or '|' in last_line),
            any(word in last_line.lower() for word in
                ['founder', 'ceo', 'cto', 'director', 'manager', 'google', 'linkedin']),
            re.match(r'^[A-Z][a-z]+,\s*', last_line),
        ]
        if any(signature_indicators):
            text = '\n'.join(lines[:-1]).strip()

    name = ''
    intro_line = ''
    middle_body = ''
    last_paragraph = ''

    name_match = re.search(r'^Hi\s+([^,!.]+)[,!.]', text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        text_after_hi = text[name_match.end():].strip()
    else:
        text_after_hi = text

    text_after_hi = ' '.join(text_after_hi.split())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', text_after_hi)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if s and len(s.strip()) > 3]

    if len(sentences) <= 1 and '.' in text_after_hi:
        parts = text_after_hi.split('. ')
        sentences = []
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                if i < len(parts) - 1:
                    part += '.'
                elif not part.endswith(('.', '!', '?')):
                    part += '.'
                sentences.append(part)
        sentences = [s for s in sentences if s and len(s.strip()) > 3]

    if len(sentences) >= 1: intro_line = sentences[0]

    # Always assign the last 2 sentences to last_paragraph for consistency
    if len(sentences) >= 5:
        middle_body = ' '.join(sentences[1:3])
        last_paragraph = ' '.join(sentences[-2:])
    elif len(sentences) == 4:
        middle_body = sentences[1]          # 1 middle sentence so last 2 fit in last_paragraph
        last_paragraph = ' '.join(sentences[-2:])
    elif len(sentences) == 3:
        middle_body = ''
        last_paragraph = ' '.join(sentences[-2:])
    elif len(sentences) == 2:
        middle_body = ''
        last_paragraph = sentences[-1]      # only 1 non-intro sentence available
    elif len(sentences) == 1:
        middle_body = ''
        last_paragraph = ''

    return {'Name': name, 'Intro Line': intro_line, 'Middle Body': middle_body,
            'Last Paragraph': reorder_last_paragraph(last_paragraph)}


# Fixed closing question pattern used in the new 1st Follow-up format
# All patterns that appear in any CTA closing line of the 1st Follow-up
FOLLOWUP_1_CLOSING_PATTERNS = [
    r'do you or your ventures outsource',
    r'do you outsource',
    r'open to a quick call to explore synergies',
    r'open to a quick call',
    r'open to a short call',
    r'open to a brief call',
    r'open to a quick conversation',
    r'worth a.*call',
]

# The single normalized last line that should appear for every row
FOLLOWUP_1_STANDARD_LAST_LINE = (
    "Do you or your ventures outsource AI or software development work? "
    "Open to a quick call to explore synergies?"
)


def parse_followup_1(followup_text):
    """
    Parse 1st Followup message (new format — March 2026).

    New structure:
        Hi [Name],
        Following up on my previous [note/mail]. [Optional 1-2 context sentences.]
        Do you or your ventures outsource AI or software development work? Open to a quick call to explore synergies?

    Fields returned:
        followup-1-intro line  – The "Following up…" opener sentence.
        followup-1-middle      – Prospect-specific context sentence(s) between the intro and closing question.
        followup-1-last line   – Always the standard closing question (normalized).
        link                   – Empty string (no link in new format; kept for schema compatibility).
    """
    if pd.isna(followup_text) or not str(followup_text).strip():
        return {'followup-1-intro line': '', 'followup-1-middle': '', 'link': '', 'followup-1-last line': ''}

    text = str(followup_text).strip()

    # Strip the greeting line ("Hi Name,")
    name_match = re.search(r'^Hi\s+[^,!.]+[,!.]', text, re.IGNORECASE)
    text_after_hi = text[name_match.end():].strip() if name_match else text

    # Flatten multi-line text into a list of non-empty stripped lines
    raw_lines = [l.strip() for l in text_after_hi.split('\n') if l.strip()]

    # If lines are very few, try to split on sentence boundaries instead
    if len(raw_lines) < 2:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_after_hi)
        raw_lines = [s.strip() for s in sentences if s.strip()]

    def _is_closing_question(line):
        """Returns True if the line matches any known CTA closing pattern."""
        ll = line.lower()
        return any(re.search(pat, ll) for pat in FOLLOWUP_1_CLOSING_PATTERNS)

    # Separate closing question from body
    closing_idx = -1
    for i, line in enumerate(raw_lines):
        if _is_closing_question(line):
            closing_idx = i
            break

    if closing_idx == -1:
        # No explicit closing question found — treat last line as closing
        closing_idx = len(raw_lines) - 1

    body_lines = raw_lines[:closing_idx]    # everything before the closing question

    # The first body line is the intro ("Following up on my previous…")
    intro_line = body_lines[0] if body_lines else ''

    # Remaining body lines are the prospect-specific middle
    middle = ' '.join(body_lines[1:]).strip() if len(body_lines) > 1 else ''

    return {
        'followup-1-intro line': intro_line,
        'followup-1-middle': middle,
        'link': '',               # No link in new format
        'followup-1-last line': FOLLOWUP_1_STANDARD_LAST_LINE,  # always normalized
    }


def parse_followup_2(followup_text):
    """
    Parse 2nd Followup message.
    """
    if pd.isna(followup_text) or not followup_text.strip():
        return {'followup-2-intro line': '', 'followup-2-middle': '', 'followup-2-last line': ''}

    text = str(followup_text).strip()
    name_match = re.search(r'^Hi\s+([^,!.]+)[,!.]', text, re.IGNORECASE)
    if name_match:
        text_no_hi = text[name_match.end():].strip()
    else:
        text_no_hi = text

    lines = [l.strip() for l in text_no_hi.split('\n') if l.strip()]
    closing_words = ['best', 'thanks', 'regards', 'sincerely', 'cheers', 'warm regards']
    valid_lines_count = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        is_signature = False
        if '|' in line and len(line) < 100:
            is_signature = True
        elif any(
            word in line.lower() for word in ['founder', 'ceo', 'cto', 'director', 'manager', 'linkedin', 'built by']):
            is_signature = True
        elif re.match(r'^[A-Z][a-z]+,\s+[A-Z]', line) and len(line) < 50:
            is_signature = True
        elif any(line.lower().startswith(word) for word in closing_words) and len(line) < 50:
            is_signature = True
        elif len(line) < 40 and line[-1] not in ['.', '!', '?', '"', "'"]:
            is_signature = True

        if is_signature:
            valid_lines_count = i
        else:
            break

    lines = lines[:valid_lines_count]
    if len(lines) < 3:
        joined_text = ' '.join(lines) if lines else text_no_hi
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', joined_text)
        lines = [s.strip() for s in sentences if s.strip()]

    intro_line = ''
    middle = ''
    last_line = ''

    if len(lines) >= 1: intro_line = lines[0]
    if len(lines) >= 3:
        last_line = lines[-1]
        middle = ' '.join(lines[1:-1])
    elif len(lines) == 2:
        last_line = lines[-1]
        middle = ''

    return {'followup-2-intro line': intro_line, 'followup-2-middle': middle, 'followup-2-last line': last_line}


try:
    # Read the CSV
    df = pd.read_csv(file_path)

    # ---------------- Parsing ----------------
    parsed_intros = df['Introductory mail'].apply(parse_intro_mail)
    df['Name'] = parsed_intros.apply(lambda x: x['Name'])
    df['Intro Line'] = parsed_intros.apply(lambda x: x['Intro Line'])
    df['Middle Body'] = parsed_intros.apply(lambda x: x['Middle Body'])
    df['Last Paragraph'] = parsed_intros.apply(lambda x: x['Last Paragraph'])

    parsed_followup1 = df['1st Followup'].apply(parse_followup_1)
    df['followup-1-intro line'] = parsed_followup1.apply(lambda x: x['followup-1-intro line'])
    df['followup-1-middle'] = parsed_followup1.apply(lambda x: x['followup-1-middle'])
    df['link'] = parsed_followup1.apply(lambda x: x['link'])
    df['followup-1-last line'] = parsed_followup1.apply(lambda x: x['followup-1-last line'])

    parsed_followup2 = df['2nd Followup'].apply(parse_followup_2)
    df['followup-2-intro line'] = parsed_followup2.apply(lambda x: x['followup-2-intro line'])
    df['followup-2-middle'] = parsed_followup2.apply(lambda x: x['followup-2-middle'])
    df['followup-2-last line'] = parsed_followup2.apply(lambda x: x['followup-2-last line'])

    # ---------------- Validation ----------------
    print('Validating email addresses (MX record lookup)...')
    validation_results = df['Email'].apply(validate_emails_field)
    df['email_valid'] = validation_results.apply(lambda x: x[0])
    df['email_status'] = validation_results.apply(lambda x: x[1])

    print('Validating links (HEAD request)...')
    link_results = df['link'].apply(validate_and_read_link)
    df['link_valid'] = link_results.apply(lambda x: x['link_valid'])
    df['link_status'] = link_results.apply(lambda x: x['link_status'])
    df['link_final_url'] = link_results.apply(lambda x: x['link_final_url'])

    # ---------------- Scoring ----------------
    if ST_AVAILABLE:
        print('Computing semantic relevance scores...')
    relevance_results = df.apply(compute_email_relevance, axis=1)
    df['relevance_score'] = relevance_results.apply(lambda x: x['relevance_score'])
    df['relevance_label'] = relevance_results.apply(lambda x: x['relevance_label'])
    df['relevance_detail'] = relevance_results.apply(lambda x: x['relevance_detail'])

    # ---------------- Formatting (Bolding/Cleaning) ----------------
    # Apply formatting row-by-row to the parsed columns in the DataFrame
    print('Applying bold formatting and HTML tags...')


    def fix_sentence_spacing(text):
        """
        Ensures a single space follows any sentence-ending punctuation (. ! ?)
        that is immediately followed by a letter — even when an HTML closing tag
        (e.g. </b>) sits between the punctuation and the next word.
        Handles cases like:
            'back in.As' -> 'back in. As'
            '</b>.We'    -> '</b>. We'
        """
        if not text:
            return text
        # After punctuation + optional closing tags, before an uppercase letter
        text = re.sub(r'([.!?])((?:</[^>]+>)*)([A-Za-z])', r'\1\2 \3', text)
        return text

    def format_row_in_place(row):
        # We need to clean and optionally bold specific text columns.

        # Columns that get general cleaning (strip chars, capitalize)
        clean_text_cols = [
            'Name', 'Intro Line', 'Middle Body', 'Last Paragraph',
            'followup-1-intro line', 'followup-1-middle', 'followup-1-last line',
            'followup-2-intro line', 'followup-2-middle', 'followup-2-last line',
            'Subject'
        ]

        for col in clean_text_cols:
            if col in row and pd.notna(row[col]):
                val = str(row[col])
                # Clean: remove quotes/slashes, strip, capitalize
                val = val.replace('"', '').replace('/', '').replace('\\', '').strip()
                # Ensure a space after sentence-ending punctuation
                val = fix_sentence_spacing(val)
                if val:
                    val = val[0].upper() + val[1:]

                # Apply Bolding Logic
                if col == 'Middle Body' and val:
                    val = add_bold_formatting(val, top_n=8)
                elif col == 'Last Paragraph' and val:
                    val = add_bold_formatting(val, top_n=4)
                elif col == 'followup-1-middle' and val:
                    val = add_bold_formatting(val, top_n=4)
                elif col == 'followup-1-last line' and val:
                    val = add_bold_formatting(val, top_n=4)
                elif col == 'followup-2-middle' and val:
                    val = add_bold_formatting(val, top_n=6)
                elif col == 'followup-2-last line' and val:
                    val = add_bold_formatting(val, top_n=4)

                row[col] = val
            else:
                row[col] = ''

        # Handle Link Formatting
        if 'link' in row and pd.notna(row['link']):
            val = str(row['link']).replace('"', '').replace('\\', '').rstrip('/').strip()
            if val:
                row['link'] = f"<a href='{val}'>here</a>"
            else:
                row['link'] = ''

        # Handle Link Final URL Formatting (Clean only)
        if 'link_final_url' in row and pd.notna(row['link_final_url']):
            val = str(row['link_final_url']).replace('"', '').replace('\\', '').rstrip('/').strip()
            row['link_final_url'] = val

        return row


    # Apply formatting to the DataFrame
    df = df.apply(format_row_in_place, axis=1)

    # ---------------- Export ----------------

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'Processed_Email_Script_Output_{timestamp}.csv'
    # Save everything to CSV
    df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print(f"SUCCESS: Data processed and saved to '{output_filename}'")
    print("The CSV includes original columns, parsed columns, validation results, and HTML formatted text.")
    print("=" * 80)

    # ---------------- JSON Preview (10 sample rows) ----------------
    # Columns to include in the JSON preview
    preview_cols = [
        'Reviewer Name', 'Email', 'Reviewer Company',
        'Name', 'Subject',
        'Intro Line', 'Middle Body', 'Last Paragraph',
        'followup-1-intro line', 'followup-1-middle', 'followup-1-last line',
        'followup-2-intro line', 'followup-2-middle', 'followup-2-last line',
        'email_valid', 'email_status',
        'relevance_score', 'relevance_label',
    ]

    # Only keep columns that actually exist in the dataframe
    available_cols = [c for c in preview_cols if c in df.columns]

    # Drop fully-empty rows and take first 10 non-empty records
    sample_df = df.dropna(how='all').head(10)[available_cols]

    print("\n" + "=" * 80)
    print("JSON PREVIEW — first 10 processed rows:")
    print("=" * 80)
    sample_json = sample_df.to_dict(orient='records')
    print(json.dumps(sample_json, indent=2, ensure_ascii=False))
    print("=" * 80)

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
except KeyError as e:
    print(f"Error: Column not found in CSV - {e}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()