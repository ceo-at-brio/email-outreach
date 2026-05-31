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

INPUT_CSV = "Email - 3rd May - 27_4.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"

BANNED_WORDS = ["curious", "excited", "keen", "proposal", "outsource", "synergy"]

SYSTEM_INSTRUCTION = """
You are Parag, Founder of SoftwareBrio. You write highly engaging, ultra-concise, and classy cold emails. 
Rule 1: NEVER output markdown code blocks (like ```text).
Rule 2: Do NOT include conversational filler like "Here is the email".
Rule 3: Output ONLY the raw email text, ready to be sent.
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
* FORMATTING: MUST start with 'Subject: ' followed by a new line for the body. Use double line breaks (\n\n) between every sentence in the body.
* BOLDING: Use HTML <b> tags around the technical headaches. Do NOT bold anything else.
* Tone: Founder-to-Founder. Casual, confident.
* Do NOT use banned words: curious, excited, keen, proposal, outsource.
"""

FOLLOWUP_1_PROMPT = """
Write an email follow-up message.

Structure:
* Start: "Hi {first_name}, I was thinking more about[Mention a specific product feature or workflow from their Company About / Project Review data]." 
* Value Drop: Provide one 1-sentence insight on how elite engineering teams tackle that specific workflow, using <b>HTML bold tags</b> around the specific technical solution (e.g., <b>caching strategies</b>).
* The Ask: "If expanding your tech bandwidth is on your radar, I'd love to share how our ex-FAANG team handles this. Worth a 10-min chat?"
* Sign-off: "Parag | Founder, SoftwareBrio.com"

Constraints:
* Under 60 words. No fluff. 
* FORMATTING: Use double line breaks (\n\n) between every sentence.
* BOLDING: Only bold the technical solution.
* Do NOT use banned words.
"""

FOLLOWUP_2_PROMPT = """
Write a final, punchy follow-up mail designed to force a reply.

Structure:
* Start: "Hi {first_name}, wrapping up my outreach here."
* The Hook: Synthesize their Company About and Live Research to pitch ONE highly specific tech/AI feature idea for their platform. Put <b>HTML bold tags</b> around the specific feature name.
* The Ask: "If you're ever looking for a reliable, ex-FAANG dev team to build features like that fast, keep us in mind. Open to a quick intro call so I can put a face to the name?"
* End exactly with: "Best, Parag (Ex-Google)"

Constraints:
* Under 50 words.
* FORMATTING: Use double line breaks (\n\n) between every sentence.
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


def research_lead_with_google(name, company, linkedin_url, retries=3):
    """Uses Gemini's native Google Search grounding with LinkedIn URL verification."""

    research_prompt = f"""
    Use Google Search to find recent news, product launches, or technical milestones for '{name}' at the company '{company}'. 

    CRITICAL IDENTITY ANCHOR:
    Their exact LinkedIn profile URL is: {linkedin_url}
    Use the data indexed at this URL to verify you are researching the correct person and company.

    STRICT RULES:
    1. Focus heavily on '{company}'. 
    2. If you find recent news specifically involving '{name}' at '{company}', include it. 
    3. WARNING: Do NOT confuse '{name}' with someone else who has the same name. If the search results do not align with the provided LinkedIn URL, DO NOT mention the person.
    4. If you cannot find recent news, just summarize the company's core product and target audience.

    Return 2-3 bullet points of highly relevant business context.
    """

    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0  # Zero creativity, strict facts only
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
    """Forces rewrites for length AND banned words."""
    raw_text = send_message_with_retry(chat_session, prompt)
    if "ERROR:" in raw_text: return raw_text
    email_text = clean_text(raw_text)

    attempts = 0
    while attempts < retries:
        word_count = len(email_text.split())
        has_banned = contains_banned_words(email_text)

        if word_count <= max_words and not has_banned:
            break  # Passes all checks!

        print(
            f"    [!] Constraint failed (Words: {word_count}/{max_words} | Banned Words: {has_banned}). Forcing rewrite...")
        correction = f"Your last response failed constraints. It MUST be under {max_words} words. It MUST NOT use any of these words: {', '.join(BANNED_WORDS)}. Keep the HTML bolding and double line breaks. Output ONLY the email."

        raw_text = send_message_with_retry(chat_session, correction)
        if "ERROR:" in raw_text: return raw_text
        email_text = clean_text(raw_text)
        attempts += 1

    return email_text


def split_subject_and_body(email_text):
    """Splits the intro email so Outreach tools can read the Subject Line."""
    if "ERROR:" in email_text:
        return "ERROR", email_text

    # Regex to find "Subject: [text]" and split the rest into the body
    match = re.search(r'(?i)^(?:subject:\s*)(.*?)(?:\r?\n)+(.*)', email_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Fallback if AI didn't write "Subject:"
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

    # Split Intro into two columns for your sending platform
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

        # --- NEW: Extract the LinkedIn URL ---
        linkedin_url = str(row.get('LinkedIn URL', '')).strip()

        print(f"\n=======================================================")
        print(f"Processing ({i}/10): {first_name} at {company}")

        print(f"[*] Running live Google Search for context (Verified via LinkedIn)...")
        # --- NEW: Pass the URL to the research function ---
        live_research = research_lead_with_google(name, company, linkedin_url)
        df_filtered.at[index, 'Live_Research_Data'] = live_research

        # Inject the live research and LinkedIn URL into the context!
        lead_context = f"CONTEXT FOR THIS LEAD:\nName: {name}\nLinkedIn: {linkedin_url}\nCompany: {company}\nPast Msg: {reachout}\nProject Review: {review}\nCompany About: {about}\n\nLIVE GOOGLE SEARCH RESEARCH:\n{live_research}"

        chat = client.chats.create(model="gemini-2.5-pro", config=config)

        # 1. Intro
        raw_intro = generate_with_constraints(
            chat,
            lead_context + "\n\n" + INTRO_PROMPT.format(company=company),
            max_words=75
        )
        subject, body = split_subject_and_body(raw_intro)
        df_filtered.at[index, 'Intro_Subject'] = subject
        df_filtered.at[index, 'Intro_Body'] = body

        # 2. Follow-up 1
        if "ERROR:" not in raw_intro:
            fu1_msg = generate_with_constraints(
                chat,
                FOLLOWUP_1_PROMPT.format(first_name=first_name),
                max_words=60
            )
        else:
            fu1_msg = "Skipped due to Intro error"
        df_filtered.at[index, 'Generated_Followup_1'] = fu1_msg

        # 3. Follow-up 2
        if "ERROR:" not in fu1_msg:
            fu2_msg = generate_with_constraints(
                chat,
                FOLLOWUP_2_PROMPT.format(first_name=first_name),
                max_words=50
            )
        else:
            fu2_msg = "Skipped due to Follow-up 1 error"
        df_filtered.at[index, 'Generated_Followup_2'] = fu2_msg

        # --- LIVE JSON CONSOLE PRINTING ---
        output_json = {
            "Lead": f"{first_name} at {company}",
            "Live_Research_Found": live_research,
            "Emails": {
                "Subject": subject,
                "Intro_Body": body,
                "Follow-up_1": fu1_msg,
                "Follow-up_2": fu2_msg
            }
        }
        print(json.dumps(output_json, indent=4, ensure_ascii=False))
        # ----------------------------------

        # Sleep 17 seconds to respect Free Tier limits
        if i < len(df_filtered):
            print(f"Waiting 17 seconds for rate limits...")
            time.sleep(17)

    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Generated POC emails saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()