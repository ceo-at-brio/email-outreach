"""
Full outreach pipeline:
  STEP 1 → Generate emails with Gemini (ai-studio-2 logic)
  STEP 2 → Validate every field per lead; retry generation up to MAX_GEN_RETRIES
  STEP 3 → Push only fully-validated leads to Apollo + enroll in sequence

Leads that fail validation after all retries are written to a separate
*_FAILED.csv so you can fix/regenerate them manually and re-run.
"""

import pandas as pd
import requests
import time
import re
import os
import json
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ============================================================
# CONFIGURATION — edit these before running
# ============================================================

# --- Gemini ---
GEMINI_API_KEY = "AIzaSyBIEGzk2a3Xxd45N2dbmx7UvIT3obilh-k"

# --- Apollo ---
APOLLO_API_KEY      = "qHZgYECHVdxOrg4bb69-oQ"
SEQUENCE_ID         = "6a1154f510c64c00105b527f"
SENDER_EMAIL_OVERRIDE = "parag@softwarebrio.in"   # set None to auto-pick

# --- Files ---
INPUT_CSV  = "Email - 3rd May - 27_4.csv"
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"
FAILED_CSV = f"Generated_Emails_FAILED_{timestamp}.csv"

# --- Pipeline knobs ---
LEAD_LIMIT       = 10    # set None to process all rows
MAX_GEN_RETRIES  = 3     # how many times to retry generation per lead if validation fails
RATE_LIMIT_SLEEP = 17    # seconds between leads (Free Tier: 15 RPM)
APOLLO_TIMEOUT   = 15    # seconds per Apollo API call

# ============================================================
# PROMPT CONFIG (from ai-studio-2.py)
# ============================================================
BANNED_WORDS = [
    "curious", "excited", "keen", "proposal", "outsource", "synergy",
    "innovative", "revolutionary", "disrupt", "guarantee", "streamline",
    "game-changer", "delve", "unlock",
]

SYSTEM_INSTRUCTION = """
You are Parag, Founder of SoftwareBrio. You write highly engaging, ultra-concise, and classy cold emails.
Rule 1: NEVER output markdown code blocks (like ```text).
Rule 2: Do NOT include conversational filler. Output ONLY the raw text.
Rule 3: Analyze the prospect's context. IF they seem highly technical (CTO/Engineering), use hard engineering concepts (e.g., CI/CD, database latency). IF they are business-focused (CEO/Founder), focus on business impact (e.g., speed-to-market, tech debt).
"""

INTRO_PROMPT = """
Write a warm, expert-led outreach email from Parag (ex-Google, Founder of SoftwareBrio).

Structure:
Subject:[2-4 words, Sentence case, internal memo style. MUST extract a specific keyword from their Live Research or Company About. Do NOT use a generic "Scaling [company]" template. Make it unique to them.]
Body:[1 casual sentence opening. IF LIVE GOOGLE SEARCH shows a recent milestone, mention it. IF NOT, compliment their core product.][The Pivot: "At Google, I saw firsthand how scaling products like yours often leads to <b>[Specific Tech Headache 1]</b> and <b>[Specific Tech Headache 2]</b>. We built SoftwareBrio with ex-Meta & Google engineers to solve exactly this."][The Ask: "Are you exploring external engineering bandwidth to accelerate your upcoming features?"][Closing: "Open to comparing notes next week?"]

Constraints:
* Under 75 words total.
* FORMATTING: MUST start with 'Subject: ' followed by a new line for the body. Use double line breaks (\\n\\n) between every sentence in the body.
* BOLDING: Use HTML <b> tags around the technical headaches. Do NOT bold anything else.
* Tone: Founder-to-Founder. Casual, confident.
* Do NOT use banned words.
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
* FORMATTING: Use double line breaks (\\n\\n) between every sentence.
* BOLDING: Only bold the technical solution.
* Do NOT invent or mention past clients.
* Do NOT use banned words.
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
* FORMATTING: Use double line breaks (\\n\\n) between every sentence.
* Do NOT use banned words.
"""

# ============================================================
# APOLLO CONFIG
# ============================================================
APOLLO_BASE_URL = "https://api.apollo.io/api/v1"
APOLLO_HEADERS  = {
    "x-api-key": APOLLO_API_KEY,
    "Content-Type": "application/json",
    "Cache-Control": "no-cache",
}
CUSTOM_FIELD_MAPPING = {
    "ai_subject":    "6a11393994a700000c18d5d7",
    "ai_intro_body": "6a113bfd66051a0020267334",
    "ai_followup_1": "6a11396c0f6b9b0014ab1f2e",
    "ai_followup_2": "6a11397cc7797c001018996a",
}

# ============================================================
# STEP 1 — EMAIL GENERATION HELPERS
# ============================================================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
gemini_config  = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7,
)


def clean_text(text):
    if not text:
        return ""
    return text.replace("```text", "").replace("```json", "").replace("```html", "").replace("```", "").strip()


def contains_banned_words(text):
    text_lower = text.lower()
    return any(word in text_lower for word in BANNED_WORDS)


def research_lead_with_google(name, company, linkedin_url, retries=3):
    research_prompt = f"""
    Use Google Search to find recent news, product launches, or technical milestones for '{name}' at '{company}'.
    CRITICAL IDENTITY ANCHOR: Their exact LinkedIn profile URL is: {linkedin_url}
    STRICT RULES:
    1. Focus heavily on '{company}'.
    2. If you find recent news specifically involving '{name}', include it.
    3. Do NOT confuse '{name}' with someone else. If results don't align with the LinkedIn URL, DO NOT mention the person.
    4. If no recent news, summarize the company's core product.
    Return 2-3 bullet points of highly relevant business context.
    """
    search_config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        temperature=0.0,
    )
    delay = 5
    for _ in range(retries):
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=research_prompt,
                config=search_config,
            )
            if resp.text:
                return resp.text.strip()
        except APIError as e:
            if e.code == 429:
                print(f"    [~] Search rate limit. Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"(Research failed: API Error {e.code})"
        except Exception as e:
            return f"(Research failed: {str(e)})"
    return "(No live data — max retries hit)"


def send_message_with_retry(chat_session, prompt, retries=3):
    delay = 5
    for _ in range(retries):
        try:
            resp = chat_session.send_message(prompt)
            if not resp.text:
                return "ERROR: Response blocked by safety filters."
            return resp.text
        except APIError as e:
            if e.code == 429:
                print(f"    [~] Chat rate limit. Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                return f"ERROR: API Error {e.code}: {e.message}"
        except Exception as e:
            return f"ERROR: {str(e)}"
    return "ERROR: Max retries exceeded."


def generate_with_constraints(chat_session, prompt, max_words, retries=3):
    raw_text = send_message_with_retry(chat_session, prompt)
    if "ERROR:" in raw_text:
        return raw_text
    email_text = clean_text(raw_text)

    for _ in range(retries):
        word_count = len(email_text.split())
        has_banned = contains_banned_words(email_text)
        if word_count <= max_words and not has_banned:
            break
        print(f"    [!] Constraint fail (words: {word_count}/{max_words}, banned: {has_banned}). Rewriting...")
        correction = (
            f"Your last response failed constraints. MUST be under {max_words} words. "
            f"MUST NOT use: {', '.join(BANNED_WORDS)}. Keep HTML formatting. Output ONLY the raw text."
        )
        raw_text = send_message_with_retry(chat_session, correction)
        if "ERROR:" in raw_text:
            return raw_text
        email_text = clean_text(raw_text)

    return email_text


def split_subject_and_body(email_text):
    if "ERROR:" in email_text:
        return "ERROR", email_text
    match = re.search(r'(?i)^(?:subject:\s*)(.*?)(?:\r?\n)+(.*)', email_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "quick question", email_text.strip()


# ============================================================
# STEP 2 — VALIDATION
# ============================================================
FIELD_RULES = {
    #  field_key          max_words  require_bold
    "Intro_Subject":     (10,        False),
    "Intro_Body":        (75,        True),
    "Generated_Followup_1": (60,     True),
    "Generated_Followup_2": (65,     True),
}


def validate_lead_emails(row):
    """
    Returns (passed: bool, issues: list[str]).
    Checks every required field for:
      - Non-empty
      - No "ERROR:" marker
      - No banned words
      - Word count within limit
      - At least one <b> tag where bolding is required
    """
    issues = []
    for field, (max_words, require_bold) in FIELD_RULES.items():
        value = str(row.get(field, "")).strip()

        if not value:
            issues.append(f"{field}: EMPTY")
            continue

        if "ERROR:" in value:
            issues.append(f"{field}: contains generation error marker")
            continue

        if "Skipped due to" in value:
            issues.append(f"{field}: was skipped due to upstream error")
            continue

        word_count = len(value.split())
        if word_count > max_words:
            issues.append(f"{field}: too long ({word_count}/{max_words} words)")

        if contains_banned_words(value):
            found = [w for w in BANNED_WORDS if w in value.lower()]
            issues.append(f"{field}: contains banned word(s): {found}")

        if require_bold and "<b>" not in value:
            issues.append(f"{field}: missing required <b> HTML bold tag")

    return (len(issues) == 0), issues


def generate_emails_for_lead(lead_context, first_name, company):
    """Runs the full 3-email generation for a single lead. Returns a dict of field→value."""
    chat = gemini_client.chats.create(model="gemini-2.5-pro", config=gemini_config)

    raw_intro = generate_with_constraints(
        chat,
        lead_context + "\n\n" + INTRO_PROMPT.format(company=company),
        max_words=75,
    )
    subject, body = split_subject_and_body(raw_intro)

    if "ERROR:" not in raw_intro:
        fu1 = generate_with_constraints(chat, FOLLOWUP_1_PROMPT.format(first_name=first_name), max_words=60)
    else:
        fu1 = "Skipped due to Intro error"

    if "ERROR:" not in fu1:
        fu2 = generate_with_constraints(chat, FOLLOWUP_2_PROMPT.format(first_name=first_name), max_words=65)
    else:
        fu2 = "Skipped due to Follow-up 1 error"

    return {
        "Intro_Subject":        subject,
        "Intro_Body":           body,
        "Generated_Followup_1": fu1,
        "Generated_Followup_2": fu2,
    }


# ============================================================
# STEP 3 — APOLLO PUSH HELPERS
# ============================================================
def clean_val(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def is_valid_email(email_str):
    if not email_str:
        return False
    try:
        if pd.isna(email_str):
            return False
    except (TypeError, ValueError):
        pass
    pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    return bool(re.match(pattern, str(email_str).strip()))


def get_active_email_account_id():
    url = f"{APOLLO_BASE_URL}/email_accounts"
    try:
        resp = requests.get(url, headers=APOLLO_HEADERS, timeout=APOLLO_TIMEOUT)
        if resp.status_code == 200:
            accounts = resp.json().get("email_accounts", [])
            active   = [a for a in accounts if a.get("active") is True]
            if not active:
                print("[!] No active Apollo inboxes found.")
                return None
            if SENDER_EMAIL_OVERRIDE:
                for acct in active:
                    if acct.get("email") == SENDER_EMAIL_OVERRIDE:
                        print(f"[+] Sender: {acct['email']} (ID: {acct['id']})")
                        return acct["id"]
                print(f"[!] SENDER_EMAIL_OVERRIDE '{SENDER_EMAIL_OVERRIDE}' not found.")
                print(f"    Available: {[a['email'] for a in active]}")
                return None
            selected = active[0]
            print(f"[+] Sender: {selected['email']} (ID: {selected['id']})")
            return selected["id"]
        print(f"[-] Failed to fetch email accounts: {resp.text}")
    except Exception as e:
        print(f"[!] Error fetching email accounts: {e}")
    return None


def find_existing_contact_by_email(email):
    url = f"{APOLLO_BASE_URL}/contacts/search"
    payload = {"q_keywords": email, "per_page": 25}

    def _search():
        resp = requests.post(url, json=payload, headers=APOLLO_HEADERS, timeout=APOLLO_TIMEOUT)
        if resp.status_code == 200:
            for contact in resp.json().get("contacts", []):
                stored = [contact.get("email", "")] + [
                    e.get("email", "") for e in contact.get("contact_emails", [])
                ]
                if email.lower() in [s.lower() for s in stored if s]:
                    return contact.get("id")
        return None

    try:
        return _search()
    except requests.exceptions.Timeout:
        print(f"    [!] Search timed out for {email}. Retrying...")
        time.sleep(2)
        try:
            return _search()
        except Exception:
            pass
    except Exception as e:
        print(f"    [!] Search error for {email}: {e}")
    return None


def push_to_apollo(lead_data, email_account_id):
    raw_email = lead_data.get("Email ID")
    if not is_valid_email(raw_email):
        print(f"    [-] Invalid email: '{raw_email}'")
        return False

    email      = str(raw_email).strip()
    name       = clean_val(lead_data.get("Reviewer Name"))
    first_name = name.split()[0] if name else "There"
    last_name  = " ".join(name.split()[1:]) if name else ""

    typed_custom_fields = {
        CUSTOM_FIELD_MAPPING["ai_subject"]:    clean_val(lead_data.get("Intro_Subject")),
        CUSTOM_FIELD_MAPPING["ai_intro_body"]: clean_val(lead_data.get("Intro_Body")),
        CUSTOM_FIELD_MAPPING["ai_followup_1"]: clean_val(lead_data.get("Generated_Followup_1")),
        CUSTOM_FIELD_MAPPING["ai_followup_2"]: clean_val(lead_data.get("Generated_Followup_2")),
    }

    existing_id = find_existing_contact_by_email(email)
    contact_id  = None

    if existing_id:
        print(f"    [*] Existing contact found (ID: {existing_id}). Patching...")
        patch_payload = {
            "first_name":          first_name,
            "last_name":           last_name,
            "organization_name":   clean_val(lead_data.get("Reviewer Company")),
            "typed_custom_fields": typed_custom_fields,
        }
        r = requests.patch(
            f"{APOLLO_BASE_URL}/contacts/{existing_id}",
            json=patch_payload, headers=APOLLO_HEADERS, timeout=APOLLO_TIMEOUT,
        )
        if r.status_code == 200:
            contact_id = existing_id
            print(f"    [+] Patched.")
        else:
            print(f"    [-] Patch failed: {r.status_code} — {r.text[:200]}")
    else:
        print(f"    [*] Creating new contact...")
        create_payload = {
            "email":               email,
            "first_name":          first_name,
            "last_name":           last_name,
            "organization_name":   clean_val(lead_data.get("Reviewer Company")),
            "typed_custom_fields": typed_custom_fields,
            "run_dedupe":          True,
        }
        r = requests.post(
            f"{APOLLO_BASE_URL}/contacts",
            json=create_payload, headers=APOLLO_HEADERS, timeout=APOLLO_TIMEOUT,
        )
        if r.status_code == 200:
            contact_id = r.json().get("contact", {}).get("id")
            print(f"    [+] Created (ID: {contact_id}).")
        else:
            print(f"    [-] Create failed: {r.status_code} — {r.text[:200]}")

    if not contact_id:
        return False

    # Enroll in sequence
    seq_url = f"{APOLLO_BASE_URL}/emailer_campaigns/{SEQUENCE_ID}/add_contact_ids"
    seq_payload = {
        "contact_ids":                    [contact_id],
        "emailer_campaign_id":            SEQUENCE_ID,
        "send_email_from_email_account_id": email_account_id,
        "async":                          False,
    }
    seq_params = {
        "sequence_unverified_email":              "true",
        "sequence_active_in_other_campaigns":     "true",
        "sequence_finished_in_other_campaigns":   "true",
    }
    sr = requests.post(seq_url, json=seq_payload, params=seq_params,
                       headers=APOLLO_HEADERS, timeout=APOLLO_TIMEOUT)

    if sr.status_code == 200:
        body    = sr.json()
        skipped = body.get("skipped_contact_ids", [])
        if contact_id in skipped:
            reason = next(
                (s.get("reason", "unknown") for s in body.get("skipped_contacts", [])
                 if s.get("id") == contact_id),
                "unknown",
            )
            print(f"    [~] Skipped by Apollo: {reason}")
        else:
            print(f"    [+] Enrolled in sequence.")
        return True

    print(f"    [-] Enrollment failed: {sr.status_code} — {sr.text[:200]}")
    return False


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 65)
    print("  OUTREACH PIPELINE")
    print("  Step 1: Generate  →  Step 2: Validate  →  Step 3: Push")
    print("=" * 65)

    # Load input CSV
    if not os.path.exists(INPUT_CSV):
        print(f"[!] Input CSV not found: {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV)
    df_filtered = df.dropna(subset=["Email ID"]).copy()
    if LEAD_LIMIT:
        df_filtered = df_filtered.head(LEAD_LIMIT)
    total = len(df_filtered)
    print(f"\n[*] Loaded {total} leads from '{INPUT_CSV}'\n")

    # Output columns
    for col in ["Intro_Subject", "Intro_Body", "Generated_Followup_1", "Generated_Followup_2", "Live_Research_Data"]:
        df_filtered[col] = ""

    validated_rows   = []
    failed_rows      = []
    validation_log   = []

    # ── STEP 1 + 2: Generate & validate per lead ─────────────────
    for i, (index, row) in enumerate(df_filtered.iterrows(), start=1):
        name       = str(row.get("Reviewer Name", "")).strip()
        first_name = name.split()[0] if name and name.lower() != "nan" else "There"
        company    = str(row.get("Reviewer Company", "")).strip()
        reachout   = str(row.get("Reach-out Message", "")).strip()
        review     = str(row.get("Review Text", "")).strip()
        about      = str(row.get("linkedin company about ", "")).strip()
        linkedin   = str(row.get("LinkedIn URL", "")).strip()
        email      = str(row.get("Email ID", "")).strip()

        print(f"\n{'─'*65}")
        print(f"  Lead {i}/{total}: {first_name} at {company} <{email}>")
        print(f"{'─'*65}")

        print(f"  [*] Researching via Google Search...")
        live_research = research_lead_with_google(name, company, linkedin)
        df_filtered.at[index, "Live_Research_Data"] = live_research

        lead_context = (
            f"CONTEXT FOR THIS LEAD:\n"
            f"Name: {name}\nLinkedIn: {linkedin}\nCompany: {company}\n"
            f"Past Msg: {reachout}\nProject Review: {review}\nCompany About: {about}\n\n"
            f"LIVE GOOGLE SEARCH RESEARCH:\n{live_research}"
        )

        best_result  = None
        best_issues  = None
        attempt      = 0

        while attempt < MAX_GEN_RETRIES:
            attempt += 1
            print(f"  [*] Generating emails (attempt {attempt}/{MAX_GEN_RETRIES})...")
            generated = generate_emails_for_lead(lead_context, first_name, company)

            # Merge into a temporary row-dict for validation
            candidate = {**row.to_dict(), **generated}
            passed, issues = validate_lead_emails(candidate)

            if passed:
                best_result = generated
                best_issues = []
                print(f"  [+] Validation PASSED.")
                break
            else:
                print(f"  [!] Validation FAILED (attempt {attempt}):")
                for issue in issues:
                    print(f"       • {issue}")
                # Keep the best (fewest issues) result as fallback
                if best_issues is None or len(issues) < len(best_issues):
                    best_result = generated
                    best_issues = issues

        # Apply whichever result we have
        for col, val in best_result.items():
            df_filtered.at[index, col] = val

        row_out = df_filtered.loc[index].to_dict()

        if not best_issues:
            validated_rows.append(row_out)
            validation_log.append({"email": email, "status": "PASS", "issues": []})
        else:
            failed_rows.append(row_out)
            validation_log.append({"email": email, "status": "FAIL", "issues": best_issues})
            print(f"  [✗] Lead FAILED after {MAX_GEN_RETRIES} attempts — will NOT be pushed to Apollo.")

        # Print generated content summary
        print(f"\n  Generated content:")
        print(f"    Subject   : {best_result.get('Intro_Subject', '')[:80]}")
        print(f"    Intro body: {len(best_result.get('Intro_Body', '').split())} words")
        print(f"    Followup1 : {len(best_result.get('Generated_Followup_1', '').split())} words")
        print(f"    Followup2 : {len(best_result.get('Generated_Followup_2', '').split())} words")

        if i < total:
            print(f"\n  Waiting {RATE_LIMIT_SLEEP}s for rate limits...")
            time.sleep(RATE_LIMIT_SLEEP)

    # Save CSVs
    if validated_rows:
        pd.DataFrame(validated_rows).to_csv(OUTPUT_CSV, index=False)
        print(f"\n[+] Validated leads saved → {OUTPUT_CSV}")
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_CSV, index=False)
        print(f"[!] Failed leads saved    → {FAILED_CSV}")

    # ── STEP 3: Apollo push for validated leads only ──────────────
    print(f"\n{'='*65}")
    print(f"  STEP 3: Pushing {len(validated_rows)} validated lead(s) to Apollo")
    print(f"{'='*65}\n")

    if not validated_rows:
        print("[!] No validated leads to push. Exiting.")
        return

    email_account_id = get_active_email_account_id()
    if not email_account_id:
        print("[!] Could not resolve sender inbox. Aborting Apollo push.")
        return

    push_success = 0
    push_failed  = 0
    for i, lead in enumerate(validated_rows, start=1):
        email = lead.get("Email ID", "")
        print(f"\n  Pushing ({i}/{len(validated_rows)}): {email}")
        ok = push_to_apollo(lead, email_account_id)
        if ok:
            push_success += 1
        else:
            push_failed += 1
        time.sleep(1)

    # ── FINAL REPORT ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*65}")
    print(f"  Total leads processed : {total}")
    print(f"  Generation passed     : {len(validated_rows)}")
    print(f"  Generation failed     : {len(failed_rows)}")
    print(f"  Apollo push succeeded : {push_success}")
    print(f"  Apollo push failed    : {push_failed}")

    if failed_rows:
        print(f"\n  Leads that need manual review ({FAILED_CSV}):")
        for entry in validation_log:
            if entry["status"] == "FAIL":
                print(f"    • {entry['email']}")
                for issue in entry["issues"]:
                    print(f"        - {issue}")

    print(f"\n  Output CSV : {OUTPUT_CSV}")
    if failed_rows:
        print(f"  Failed CSV : {FAILED_CSV}")
    print()


if __name__ == "__main__":
    main()
