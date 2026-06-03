import pandas as pd
import time
import os
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError
import config

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
client = genai.Client(api_key=config.GEMINI_API_KEY)

INPUT_CSV = "CLUTCH_LEAD_GEN - 4_5.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"

# Consistent sign-off appended to every email programmatically.
# "LinkedIn" is a hyperlink — keep it as an HTML anchor so Apollo renders it clickable.
SIGNATURE = (
    'Parag | <a href="https://www.linkedin.com/in/parag-google/">LinkedIn</a>\n'
    'Ex-Google | Founder SoftwareBrio.com'
)

# EXPANDED DELIVERABILITY ARMOR
BANNED_WORDS = [
    "curious", "excited", "keen", "proposal", "outsource", "synergy",
    "innovative", "revolutionary", "disrupt", "guarantee", "streamline",
    "game-changer", "delve", "unlock"
]

# ROLE-BASED JARGON LOGIC
SYSTEM_INSTRUCTION = """
You are Parag, Founder of SoftwareBrio. You write highly engaging, ultra-concise, and classy cold emails. 
Rule 1: NEVER output markdown code blocks (like ```text).
Rule 2: Do NOT include conversational filler. Output ONLY the raw text.
Rule 3: Adapt ALL language to the reader's technical level:
   - IF the prospect is highly technical (CTO, engineering lead, dev-tooling/infra company): use precise engineering concepts (e.g. CI/CD, database latency, event-driven sync).
   - IF the prospect is NON-technical (CEO/founder of a non-software business, marketing, operations, healthcare, trades): use ONLY plain outcomes they feel day-to-day (e.g. hours saved each week, faster launches, less manual data entry, more revenue, smoother customer experience). Do NOT use engineering jargon — this explicitly includes "tech debt", "architecture", "latency", "API", "pipeline". A dairy marketer or a therapist must understand every word.
"""

config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7
)

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================
INTRO_PROMPT = """
Write a warm, expert-led outreach email from Parag (ex-Google, Founder of SoftwareBrio).

Structure:
Subject:[2-4 words, Sentence case, internal memo style. MUST extract a specific keyword from their Live Research or Company About. Do NOT use a generic "Scaling [company]" template.]

Body:
[GREETING: "Hello {first_name}," on its own line.]

[SENTENCE 1 — Opening: 1 casual sentence. IF Live Research shows a recent milestone, reference it. IF NOT, compliment a specific product feature by name.]

[SENTENCE 2 — The Pivot: Start with "At Google, I saw firsthand how [NAME their specific type of platform, e.g. 'innovation portfolio tools', 'virtual try-on marketplaces', 'legal-tech member portals'] often hits <b>[Specific Tech Headache 1 tied to their actual product]</b> and <b>[Specific Tech Headache 2 tied to their actual product]</b>." Do NOT use "products like yours" — name the category.]

[SENTENCE 3 — Credential (ONE sentence, a STATEMENT — NO question, NO ask): Tie SoftwareBrio's ex-Google background directly to the specific challenge you just named, and connect it to their goal. End as a confident statement. E.g. "That's exactly the layer our ex-Google team is built for as [Company] scales [specific goal]." Do NOT put an ask or a question mark here — the closing line is the only ask.]

[SENTENCE 4 — Closing (the ONLY ask in the whole email): "Open to comparing notes?"]

Constraints:
* The email must contain EXACTLY ONE question mark (the closing "Open to comparing notes?"). The credential sentence must NOT end in a question.
* Aim for 290 characters for the body, hard maximum 330 (subject excluded, greeting "Hello [Name]," counts). Output shorter rather than longer.
* FORMATTING: MUST start with 'Subject: ' on its own line, then a blank line, then the body. Use double line breaks (\\n\\n) between every sentence.
* BOLDING: HTML <b> tags ONLY around the two tech headaches. Nothing else.
* Tone: Founder-to-Founder. Casual, confident, never salesy.
* Do NOT use banned words.
* Do NOT use the phrases "products like yours", "external engineering bandwidth", "upcoming features", or "solve exactly this" — these are banned as too generic.
"""

FOLLOWUP_1_PROMPT = """
Write a value-FIRST follow-up email. The goal is to build TRUST by giving a useful insight or proof of competence — NOT to re-pitch or diagnose the prospect's problems for them.

DISTINCT ROLE OF THIS EMAIL: This follow-up is about something they ALREADY HAVE TODAY — an existing feature/workflow — and the value is a CONCRETE DELIVERABLE you could hand over (a sketch, an example, a fix). Do NOT pitch a brand-new future feature here — that is the job of the FINAL follow-up. Keep this one grounded in their current product.

Structure:
* Start: "Hello {first_name}, following up on [name one SPECIFIC EXISTING product feature or workflow from their context — not a generic concept]." Keep it specific to their actual product.
* Value Drop (the key sentence): Give ONE concrete, useful thing. Choose the best fit:
    - LIGHT INDUSTRY PROOF (preferred): frame it as a pattern you've seen across their industry — e.g. "Teams in [their industry] that move to [the approach] usually see [a light, believable benefit]." Keep the benefit QUALITATIVE or loosely-quantified (e.g. "noticeably fewer support tickets", "hours back each week", "a meaningful lift in conversions") — do NOT cite a fake precise statistic or pretend it's a SoftwareBrio client result. This is an honest industry observation, not a case study.
    - A specific OUTCOME framing — the tangible result of doing this right (e.g. "idle time at terminals drops noticeably"). Lead with the result, not the architecture.
    - OR a genuine free insight about THEIR specific product they could act on.
  Put <b>HTML bold tags</b> around the single most important phrase (the benefit, result, or — only if they are technical — the solution name).
* CRITICAL — Match the language to the reader (see system instruction):
    - IF the prospect is technical (CTO, engineering, dev-tooling, infra): you MAY use precise engineering terms (e.g. event-driven sync, hermetic build containers) and frame value as a technical result ("cut CI times", "reproducible across platforms").
    - IF the prospect is non-technical (marketing, operations, founder of a non-software business): use PLAIN business language about what their team feels day-to-day. Do NOT use engineering jargon like "event-driven architecture", "API latency", "headless CMS" — they will not understand it.
* Do NOT assert "the main scaling trap is X" or "the modern fix is Y" — this presumes you know their problem better than they do and reads as formulaic. Phrase value as something you've SEEN work, not a diagnosis of their flaws.
* The Ask: ONE sentence offering something LOWER-friction than a call. VARY it — do NOT always ask for a call (the intro already did). Prefer offering a concrete artifact: "Want a quick 1-page sketch of how we'd build it?", "Want a before/after example?", "Want me to send the rough architecture? Takes 2 mins to read." Reference their product/company by name.
* Do NOT add any sign-off or signature line — it will be appended automatically.

Constraints:
* Aim for 285 characters, hard maximum 320. Output shorter rather than longer. No fluff.
* FORMATTING: Use double line breaks (\\n\\n) between every sentence.
* BOLDING: Bold exactly one phrase (the result, deliverable, or technical solution name).
* Do NOT invent or mention past clients or specific client names. You MAY say "we've seen" or "we've done this on similar systems" without naming anyone.
* Do NOT use the formulaic phrases "scaling trap", "the modern fix is", "I was thinking more about", "tech bandwidth", or "on your radar" — these are banned as too generic/repetitive.
* Do NOT use banned words.
"""

FOLLOWUP_2_PROMPT = """
Write a final follow-up mail (a graceful "breakup" email) designed to force a reply. It drops ONE sharp, FORWARD-LOOKING idea, then gracefully exits.

DISTINCT ROLE OF THIS EMAIL: Unlike the earlier follow-up (which was about their CURRENT product), this one looks to their FUTURE / roadmap — a "where you could take this next" idea — and then bows out. Do NOT repeat a present-tense fix; this is a forward-looking parting thought plus a clean exit. Do NOT open with "following up on [feature]" (that was the previous email's opener).

Structure:
* Start: A short breakup-style opener that signals this is the LAST email. e.g. "Hello {first_name}, I'll get out of your inbox after this —" or "Hello {first_name}, last idea before I close this out —". Do NOT reuse "wrapping up my outreach here" and do NOT mirror the previous email's "following up on…".
* The Idea (the key sentence — keep it SHORT, ~90 chars): Drop ONE concrete, FORWARD-LOOKING idea for where THEIR product could go next — framed as "down the road, the biggest lever is X" or "the next step we'd be excited to build is Y". A sharp peer's parting tip about their FUTURE, NOT a present-day fix and NOT a moonshot. Bold the single most important phrase in <b>HTML bold tags</b>.
* CRITICAL — Do NOT default to "AI". Most ideas should be a smart, practical improvement that is NOT AI. Only suggest an AI/ML feature if it is genuinely the obvious best fit for that specific product — and even then, describe the OUTCOME, not the buzzword. Banned framings: "AI-powered", "AI-driven", "AI model", "LLM-powered" as the headline of the idea.
* Match the language to the reader (see system instruction): technical recipients get precise terms; non-technical recipients (marketing, ops, non-software founders) get plain business language about what their team feels day-to-day.
* The Ask: ONE SHORT, low-pressure clause — e.g. "Worth a look? We'd love to build it." VARY it; do NOT reuse "our ex-Google team would love to tackle it" every time. Humble and easy to say yes/no to.
* Anti-Ghosting close (write it fresh each time, keep it SHORT ~55-65 chars): A single brief line that BOTH (a) gives permission to say no, AND (b) uses the "close my file" finality that nudges a reply. Vary the wording — e.g. "If the timing's off, just say so and I'll close your file." or "No worries if now's not the time — a quick no and I'll close my file." Do NOT write the long version "If you're completely locked in on dev bandwidth right now, totally fine—just let me know so I can close my file!" — that is too long.
* Do NOT add any sign-off or signature line — it will be appended automatically.

Constraints:
* Aim for 250 characters, hard maximum 290. Output shorter rather than longer. Every sentence must be tight.
* FORMATTING: Use double line breaks (\\n\\n) between every sentence.
* BOLDING: Bold exactly one phrase in the idea sentence.
* Do NOT invent or name specific past clients. "We've seen" / "we usually see" is fine.
* Do NOT use the formulaic phrases "wrapping up my outreach here", "our ex-Google team would love to tackle it", "our ex-FAANG team", the long "completely locked in on dev bandwidth" close, or lead the idea with "AI-powered"/"AI-driven" — these are banned as too generic/repetitive.
* Do NOT use banned words.
"""

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    if not text:
        return ""
    return text.replace("```text", "").replace("```json", "").replace("```html", "").replace("```", "").strip()

def contains_banned_words(text):
    text_lower = text.lower()
    return any(word in text_lower for word in BANNED_WORDS)

def research_lead_with_google(name, company, linkedin_url, retries=5):
    """Uses Gemini's native Google Search grounding with LinkedIn URL verification."""
    research_prompt = f"""
    Use Google Search to find recent news, product launches, or technical milestones for '{name}' at the company '{company}'. 
    CRITICAL IDENTITY ANCHOR: Their exact LinkedIn profile URL is: {linkedin_url}
    Use the data indexed at this URL to verify you are researching the correct person.

    STRICT RULES:
    1. Focus heavily on '{company}'. 
    2. If you find recent news specifically involving '{name}', include it. 
    3. WARNING: Do NOT confuse '{name}' with someone else. If the search results do not align with the LinkedIn URL, DO NOT mention the person.
    4. If you cannot find recent news, summarize the company's core product.

    Return 2-3 bullet points of highly relevant business context.
    """
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0
    )
    delay = 15
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=research_prompt,
                config=search_config
            )
            if response.text:
                return response.text.strip()
        except APIError as e:
            if e.code == 429:
                print(f"    [~] Search Rate limit hit (429). Waiting {delay}s (attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
            else:
                return f"(Live research failed: API Error {e.code})"
        except Exception as e:
            return f"(Live research failed: {str(e)})"
    return "(No recent live data found - Max retries hit)"

def send_message_with_retry(chat_session, prompt, retries=6):
    """
    Send a chat message with exponential backoff for 429 rate-limit errors.

    retries=6 gives waits of 15 → 30 → 60 → 120 → 240 → 480s (total ~15 min).
    This covers both per-minute limits (recover in <60s) and per-hour limits
    (recover in <60 min). The free tier Gemini 2.5 Pro quota is ~50 RPD; once
    that's hit, nothing will recover until the next UTC day — the pipeline will
    then exhaust retries and mark leads as ERROR (expected behaviour).
    """
    delay = 15  # start at 15s (gives RPM limits room to breathe)
    for attempt in range(retries):
        try:
            response = chat_session.send_message(prompt)
            if not response.text:
                return "ERROR: Response blocked by safety filters."
            return response.text
        except APIError as e:
            if e.code == 429:
                print(f"    [~] Chat Rate limit hit (429). Waiting {delay}s (attempt {attempt+1}/{retries})...")
                time.sleep(delay)
                delay = min(delay * 2, 480)  # cap at 8 minutes
            else:
                return f"ERROR: API Error {e.code}: {e.message}"
        except Exception as e:
            return f"ERROR: {str(e)}"
    return "ERROR: Max retries exceeded."


def generate_with_constraints(chat_session, prompt, max_chars, retries=3):
    """
    Generate email text and enforce banned-word constraints only.

    Character count is no longer a retry trigger — char overages are handled
    gracefully — char overages are purely informational now. Only genuinely
    broken outputs (banned words) cause a Gemini rewrite API call.
    """
    raw_text = send_message_with_retry(chat_session, prompt)
    if "ERROR:" in raw_text: return raw_text
    email_text = clean_text(raw_text)

    attempts = 0
    while attempts < retries:
        has_banned = contains_banned_words(email_text)

        if not has_banned:
            break

        print(f"    [!] Banned word found. Forcing rewrite...")
        correction = f"Your last response used one or more banned words. It MUST NOT use any of these: {', '.join(BANNED_WORDS)}. Keep HTML formatting if applicable. Output ONLY the raw text."

        raw_text = send_message_with_retry(chat_session, correction)
        if "ERROR:" in raw_text: return raw_text
        email_text = clean_text(raw_text)
        attempts += 1

    return email_text

def split_subject_and_body(email_text):
    if "ERROR:" in email_text:
        return "ERROR", email_text
    match = re.search(r'(?i)^(?:subject:\s*)(.*?)(?:\r?\n)+(.*)', email_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "quick question", email_text.strip()


def generate_for_lead(lead_context, first_name, company):
    """
    Generate all 3 emails for a single lead using a shared chat session.
    Returns a dict with keys: Intro_Subject, Intro_Body, Generated_Followup_1, Generated_Followup_2.
    Callable as a module function from the pipeline without running main().
    """
    chat = client.chats.create(model="gemini-2.5-pro", config=config)

    raw_intro = generate_with_constraints(
        chat, lead_context + "\n\n" + INTRO_PROMPT.format(company=company, first_name=first_name), max_chars=450
    )
    subject, body = split_subject_and_body(raw_intro)

    if "ERROR:" not in raw_intro:
        fu1_msg = generate_with_constraints(chat, FOLLOWUP_1_PROMPT.format(first_name=first_name), max_chars=320)
    else:
        fu1_msg = "Skipped due to Intro error"

    if "ERROR:" not in fu1_msg:
        fu2_msg = generate_with_constraints(chat, FOLLOWUP_2_PROMPT.format(first_name=first_name), max_chars=290)
    else:
        fu2_msg = "Skipped due to Follow-up 1 error"

    # Append the standard signature to every email body.
    # Done here (not in the prompt) so it is always consistent and correctly formatted.
    def _sign(text):
        if not text or "ERROR:" in text or "Skipped due to" in text:
            return text
        return text + "\n\n" + SIGNATURE

    return {
        "Intro_Subject":        subject,
        "Intro_Body":           _sign(body),
        "Generated_Followup_1": _sign(fu1_msg),
        "Generated_Followup_2": _sign(fu2_msg),
    }


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}")
        return

    df_filtered = df.dropna(subset=['Email ID']).copy()
    df_filtered = df_filtered.head(10)  # POC LIMIT

    print(f"Found {len(df_filtered)} valid leads for the POC. Starting generation...\n")

    df_filtered['Intro_Subject'] = ""
    df_filtered['Intro_Body'] = ""
    df_filtered['Generated_Followup_1'] = ""
    df_filtered['Generated_Followup_2'] = ""
    df_filtered['Live_Research_Data'] = ""

    for i, (index, row) in enumerate(df_filtered.iterrows(), start=1):

        name = str(row.get('Reviewer Name', '')).strip()
        first_name = name.split()[0] if name and name.lower() != 'nan' else "There"
        company = str(row.get('Reviewer Company', '')).strip()
        reachout = str(row.get('Reach-out Message', '')).strip()
        review = str(row.get('Review Text', '')).strip()
        about = str(row.get('linkedin company about ', '')).strip()
        linkedin_url = str(row.get('LinkedIn URL', '')).strip()

        print(f"\n=======================================================")
        print(f"Processing ({i}/10): {first_name} at {company}")

        print(f"[*] Running live Google Search for context (Verified via LinkedIn)...")
        live_research = research_lead_with_google(name, company, linkedin_url)
        df_filtered.at[index, 'Live_Research_Data'] = live_research

        lead_context = (
            f"CONTEXT FOR THIS LEAD:\nName: {name}\nLinkedIn: {linkedin_url}\nCompany: {company}\n"
            f"Past Msg: {reachout}\nProject Review: {review}\nCompany About: {about}\n\n"
            f"LIVE GOOGLE SEARCH RESEARCH:\n{live_research}"
        )

        generated = generate_for_lead(lead_context, first_name, company)

        df_filtered.at[index, 'Intro_Subject']        = generated['Intro_Subject']
        df_filtered.at[index, 'Intro_Body']           = generated['Intro_Body']
        df_filtered.at[index, 'Generated_Followup_1'] = generated['Generated_Followup_1']
        df_filtered.at[index, 'Generated_Followup_2'] = generated['Generated_Followup_2']

        # --- LIVE JSON CONSOLE PRINTING ---
        print(json.dumps({
            "Lead": f"{first_name} at {company}",
            "Live_Research_Found": live_research,
            "Messages": {
                "Subject":    generated['Intro_Subject'],
                "Intro_Body": generated['Intro_Body'],
                "Follow-up_1": generated['Generated_Followup_1'],
                "Follow-up_2": generated['Generated_Followup_2'],
            }
        }, indent=4, ensure_ascii=False))
        # ----------------------------------

        # Sleep 17 seconds to safely handle 4 API calls per lead on Free Tier (15 RPM limit)
        if i < len(df_filtered):
            print(f"Waiting 17 seconds for rate limits...")
            time.sleep(17)

    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Generated POC emails saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()