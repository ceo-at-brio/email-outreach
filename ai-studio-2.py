import pandas as pd
import time
import os
import json
import re
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
API_KEY = "AIzaSyB5qqAgwZt9VtiW0L9pIt_nG9ZOolTE7mg"
client = genai.Client(api_key=API_KEY)

INPUT_CSV = "Email - 3rd May - 27_4.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"

# EXPANDED DELIVERABILITY ARMOR
BANNED_WORDS = [
    "curious", "excited", "keen", "proposal", "outsource", "synergy",
    "innovative", "revolutionary", "disrupt", "guarantee", "streamline",
    "game-changer", "delve", "unlock"
]

# ADDED ROLE-BASED JARGON LOGIC
SYSTEM_INSTRUCTION = """
You are Parag, Founder of SoftwareBrio. You write highly engaging, ultra-concise, and classy cold emails. 
Rule 1: NEVER output markdown code blocks (like ```text).
Rule 2: Do NOT include conversational filler. Output ONLY the raw text.
Rule 3: Analyze the prospect's context. IF they seem highly technical (CTO/Engineering), use hard engineering concepts (e.g., CI/CD, database latency). IF they are business-focused (CEO/Founder), focus on business impact (e.g., speed-to-market, tech debt).
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
Subject:[2-4 words, lowercase, internal memo style (e.g., "scaling {company}")]
Body:[1 casual sentence opening. IF LIVE GOOGLE SEARCH shows a recent milestone, mention it. IF NOT, compliment their core product.][The Pivot: "At Google, I saw firsthand how scaling products like yours often leads to <b>[Specific Tech Headache 1]</b> and <b>[Specific Tech Headache 2]</b>. We built SoftwareBrio with ex-Meta & Google engineers to solve exactly this."][The Ask: "Are you exploring external engineering bandwidth to accelerate your upcoming features?"][Closing: "Open to comparing notes next week?"]

Constraints:
* Under 75 words total.
* FORMATTING: MUST start with 'Subject: ' followed by a new line for the body. Use double line breaks (\n\n) between every sentence.
* BOLDING: Use HTML <b> tags around the technical headaches. Do NOT bold anything else.
* Tone: Founder-to-Founder. Casual, confident.
"""

FOLLOWUP_1_PROMPT = """
Write a value-driven follow-up email.

Structure:
* Start: "Hi {first_name}, I was thinking more about[Mention a specific product feature or workflow from context]." 
* Value Drop: Provide one 1-sentence insight identifying a common scaling trap related to that workflow, and state the modern technical solution our engineering team uses to solve it. Put <b>HTML bold tags</b> around the specific technical solution.
* The Ask: "If expanding your tech bandwidth is on your radar, I'd love to share how our ex-FAANG team approaches this. Worth a 10-min chat?"
* Sign-off: "Parag | Founder, SoftwareBrio.com"

Constraints:
* Under 60 words. No fluff. 
* FORMATTING: Use double line breaks (\n\n) between every sentence.
* BOLDING: Only bold the technical solution.
* Do NOT invent or mention past clients.
"""

FOLLOWUP_2_PROMPT = """
Write a final follow-up mail designed to force a reply (includes anti-ghosting).

Structure:
* Start: "Hi {first_name}, wrapping up my outreach here."
* The Hook: Pitch ONE highly specific tech/AI feature idea for their platform. Put <b>HTML bold tags</b> around the specific feature name.
* The Ask & Anti-Ghosting: "If you're ever looking for a reliable, ex-FAANG dev team to build features like that, keep us in mind. If you're completely locked in on dev bandwidth right now, totally fine—just let me know so I can close my file!"
* End exactly with: "Best, Parag (Ex-Google)"

Constraints:
* Under 65 words.
* FORMATTING: Use double line breaks (\n\n) between every sentence.
"""

# NEW: LINKEDIN DM PROMPT
LINKEDIN_DM_PROMPT = """
Write a highly-customized LinkedIn connection request note from Parag to the prospect.

Structure:
"Hi {first_name}, saw the recent [Insert specific milestone or feature from LIVE GOOGLE SEARCH or Company About]. As an ex-Google founder running a dev agency, I love the architecture you’re building. Would love to connect! - Parag"

Constraints:
* MUST be strictly under 250 characters (LinkedIn limit).
* Output ONLY the note. No subject lines, no markdown.
* Tone: Casual, professional peer.
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


def research_lead_with_google(name, company, linkedin_url, retries=3):
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
    delay = 5
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
                print(f"    [~] Search Rate limit hit (429). Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"(Live research failed: API Error {e.code})"
        except Exception as e:
            return f"(Live research failed: {str(e)})"
    return "(No recent live data found - Max retries hit)"


def send_message_with_retry(chat_session, prompt, retries=3):
    delay = 5
    for attempt in range(retries):
        try:
            response = chat_session.send_message(prompt)
            if not response.text:
                return "ERROR: Response blocked by safety filters."
            return response.text
        except APIError as e:
            if e.code == 429:
                print(f"    [~] Chat Rate limit hit (429). Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"ERROR: API Error {e.code}: {e.message}"
        except Exception as e:
            return f"ERROR: {str(e)}"
    return "ERROR: Max retries exceeded."


def generate_with_constraints(chat_session, prompt, max_words, retries=3):
    raw_text = send_message_with_retry(chat_session, prompt)
    if "ERROR:" in raw_text: return raw_text
    email_text = clean_text(raw_text)

    attempts = 0
    while attempts < retries:
        word_count = len(email_text.split())
        has_banned = contains_banned_words(email_text)

        if word_count <= max_words and not has_banned:
            break

        print(f"    [!] Constraint failed (Words: {word_count}/{max_words} | Banned: {has_banned}). Forcing rewrite...")
        correction = f"Your last response failed constraints. It MUST be under {max_words} words. It MUST NOT use any of these words: {', '.join(BANNED_WORDS)}. Keep HTML formatting if applicable. Output ONLY the raw text."

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
    df_filtered['Generated_LinkedIn_DM'] = ""  # NEW COLUMN
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

        lead_context = f"CONTEXT FOR THIS LEAD:\nName: {name}\nLinkedIn: {linkedin_url}\nCompany: {company}\nPast Msg: {reachout}\nProject Review: {review}\nCompany About: {about}\n\nLIVE GOOGLE SEARCH RESEARCH:\n{live_research}"

        chat = client.chats.create(model="gemini-2.5-pro", config=config)

        # 1. Intro Email
        raw_intro = generate_with_constraints(chat, lead_context + "\n\n" + INTRO_PROMPT.format(company=company),
                                              max_words=75)
        subject, body = split_subject_and_body(raw_intro)
        df_filtered.at[index, 'Intro_Subject'] = subject
        df_filtered.at[index, 'Intro_Body'] = body

        # 2. Follow-up 1
        if "ERROR:" not in raw_intro:
            fu1_msg = generate_with_constraints(chat, FOLLOWUP_1_PROMPT.format(first_name=first_name), max_words=60)
        else:
            fu1_msg = "Skipped due to Intro error"
        df_filtered.at[index, 'Generated_Followup_1'] = fu1_msg

        # 3. Follow-up 2
        if "ERROR:" not in fu1_msg:
            fu2_msg = generate_with_constraints(chat, FOLLOWUP_2_PROMPT.format(first_name=first_name), max_words=65)
        else:
            fu2_msg = "Skipped due to Follow-up 1 error"
        df_filtered.at[index, 'Generated_Followup_2'] = fu2_msg

        # 4. LinkedIn DM (NEW)
        if "ERROR:" not in raw_intro:
            linkedin_msg = generate_with_constraints(chat, LINKEDIN_DM_PROMPT.format(first_name=first_name),
                                                     max_words=45)  # 45 words is safely under 300 chars
        else:
            linkedin_msg = "Skipped due to Intro error"
        df_filtered.at[index, 'Generated_LinkedIn_DM'] = linkedin_msg

        # --- LIVE JSON CONSOLE PRINTING ---
        output_json = {
            "Lead": f"{first_name} at {company}",
            "Live_Research_Found": live_research,
            "Messages": {
                "Subject": subject,
                "Intro_Body": body,
                "Follow-up_1": fu1_msg,
                "Follow-up_2": fu2_msg,
                "LinkedIn_DM": linkedin_msg
            }
        }
        print(json.dumps(output_json, indent=4, ensure_ascii=False))
        # ----------------------------------

        # Sleep 21 seconds to safely handle 5 API calls per lead on Free Tier (15 RPM limit)
        if i < len(df_filtered):
            print(f"Waiting 21 seconds for rate limits...")
            time.sleep(21)

    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Generated POC emails saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()