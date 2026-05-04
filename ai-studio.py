import pandas as pd
import time
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
API_KEY = "AIzaSyBiup-AaqYKz1EKDPX7v7oTepwVaL3Jduc"

# Initialize the NEW Google GenAI client
client = genai.Client(api_key=API_KEY)

INPUT_CSV = "Email - 3rd May - 27_4.csv"

# Add dynamic timestamp to the output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"

SYSTEM_INSTRUCTION = """
You are Parag, Founder of SoftwareBrio. You write highly engaging, ultra-concise, and classy cold emails. 
Rule 1: NEVER output markdown code blocks (like ```text).
Rule 2: Do NOT include conversational filler like "Here is the email".
Rule 3: Output ONLY the raw email text, ready to be sent.
"""

# In the new SDK, system instructions are passed via a config object
config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
)

# ==========================================
# 2. PROMPT TEMPLATES
# ==========================================
INTRO_PROMPT = """
Write a warm, expert-led outreach email using this structure:
Subject Line: A short 'Google Hook' (e.g., 'Google-scale systems for {company}').
First Para: 1-sentence genuine compliment based on their work/company. Then use this logic: 'At Google, I saw how fast growth usually leads to [Headache 1] and[Headache 2]. Those are exactly the kinds of technical hurdles we solve at SoftwareBrio.' (Tailor the headaches to their specific profile).
Second Para: Offer a free tech proposal to simplify operations.
Closing: Ask for a brief intro meeting in the coming days.

Constraints:
* MUST be strictly under 350 characters total.
* No sales pitch, no links, no words like "curious", "excited", or "keen".
* Feel peer-to-peer.
"""

FOLLOWUP_1_PROMPT = """
Write an email follow-up message strictly under 300 characters.
Start with "Hi {first_name}, Following up on my previous mail." (Do not praise them).
Share a unique insight/tip relevant to their profile. Briefly mention how we can help.
Ask: "Do you or your ventures outsource AI or Software development work?"
Ask for a short call to explore synergies.
Sign off exactly as: "Parag | Linkedin Founder, SoftwareBrio.com Built by ex-Meta & Google engineers"

Constraints:
* Strictly under 300 characters.
* Short, classy, optimized for mobile screens.
"""

FOLLOWUP_2_PROMPT = """
Write an engaging 2nd follow-up mail to force a reply.
Start with: "Hi {first_name}, Hope you're doing well! Following up on my previous mail!"
Take an idea from their profile and state specifically how SoftwareBrio can actually help them. Tie it back to the previous messages so it proves we are noticing and not AI spamming.
Ask for a 15-min call—no strings attached.
End exactly with: "Best, Parag, Google | Founder, SoftwareBrio.com"

Constraints:
* Strictly less than 260 characters.
* Immediate value, highly relevant to their specific work.
"""


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def clean_text(text):
    """Removes markdown blocks if the AI accidentally includes them."""
    if not text:
        return ""
    return text.replace("```text", "").replace("```json", "").replace("```html", "").replace("```", "").strip()


def send_message_with_retry(chat_session, prompt, retries=3):
    """Handles API Rate Limits safely using the new SDK's error handling."""
    delay = 5
    for attempt in range(retries):
        try:
            response = chat_session.send_message(prompt)
            if not response.text:
                return "ERROR: Response blocked by safety filters."
            return response.text

        except APIError as e:
            # Code 429 means "Too Many Requests" (Rate Limit)
            if e.code == 429:
                print(f"    [~] Rate limit hit (429). Waiting {delay} seconds before retrying...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff (5s, 10s, 20s)
            else:
                return f"ERROR: API Error {e.code}: {e.message}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    return "ERROR: Max retries exceeded."


def generate_with_length_check(chat_session, prompt, max_length, retries=2):
    """Forces the AI to rewrite the email if it exceeds the character limit."""
    raw_text = send_message_with_retry(chat_session, prompt)

    if "ERROR:" in raw_text:
        return raw_text

    email_text = clean_text(raw_text)

    attempts = 0
    while len(email_text) > max_length and attempts < retries:
        print(f"    [!] Length {len(email_text)} exceeds {max_length}. Forcing rewrite...")
        correction_prompt = f"Your last response was {len(email_text)} characters. It MUST be strictly under {max_length} characters. Cut words to make it shorter without losing the core meaning. Output ONLY the email."

        raw_text = send_message_with_retry(chat_session, correction_prompt)
        if "ERROR:" in raw_text:
            return raw_text

        email_text = clean_text(raw_text)
        attempts += 1

    return email_text


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}")
        return

    # Filter out invalid emails and grab the first 10
    df_filtered = df.dropna(subset=['Email ID']).copy()
    df_filtered = df_filtered.head(10)

    print(f"Found {len(df_filtered)} valid leads for the POC. Starting generation...\n")

    df_filtered['Generated_Intro'] = ""
    df_filtered['Generated_Followup_1'] = ""
    df_filtered['Generated_Followup_2'] = ""

    for i, (index, row) in enumerate(df_filtered.iterrows(), start=1):

        name = str(row.get('Reviewer Name', '')).strip()
        first_name = name.split()[0] if name and name.lower() != 'nan' else "There"
        company = str(row.get('Reviewer Company', '')).strip()
        reachout = str(row.get('Reach-out Message', '')).strip()
        review = str(row.get('Review Text', '')).strip()
        about = str(row.get('linkedin company about ', '')).strip()

        print(f"\n=======================================================")
        print(f"Processing ({i}/10): {first_name} at {company}")
        print(f"=======================================================")

        lead_context = f"CONTEXT FOR THIS LEAD -> Name: {name}, Company: {company}, Past Msg: {reachout}, Project Review: {review}, Company About: {about}."

        # Start a fresh chat using the new SDK format
        chat = client.chats.create(model="gemini-2.5-pro", config=config)

        # 1. Intro
        intro_msg = generate_with_length_check(
            chat,
            lead_context + "\n\n" + INTRO_PROMPT.format(company=company),
            max_length=350
        )
        df_filtered.at[index, 'Generated_Intro'] = intro_msg

        # 2. Follow-up 1
        if "ERROR:" not in intro_msg:
            fu1_msg = generate_with_length_check(
                chat,
                FOLLOWUP_1_PROMPT.format(first_name=first_name),
                max_length=300
            )
        else:
            fu1_msg = "Skipped due to Intro error"
        df_filtered.at[index, 'Generated_Followup_1'] = fu1_msg

        # 3. Follow-up 2
        if "ERROR:" not in fu1_msg:
            fu2_msg = generate_with_length_check(
                chat,
                FOLLOWUP_2_PROMPT.format(first_name=first_name),
                max_length=260
            )
        else:
            fu2_msg = "Skipped due to Follow-up 1 error"
        df_filtered.at[index, 'Generated_Followup_2'] = fu2_msg

        # --- LIVE JSON CONSOLE PRINTING ---
        output_json = {
            "Lead": f"{first_name} at {company}",
            "Emails": {
                "Intro": intro_msg,
                "Follow-up_1": fu1_msg,
                "Follow-up_2": fu2_msg
            }
        }
        print(json.dumps(output_json, indent=4, ensure_ascii=False))
        # ----------------------------------

        # Sleep 12 seconds to respect Gemini Free Tier 15 RPM limits
        if i < len(df_filtered):  # No need to sleep after the very last item
            print(f"Waiting 12 seconds for rate limits...")
            time.sleep(12)

    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done! Generated POC emails saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()