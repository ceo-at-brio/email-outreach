import pandas as pd
import requests
import time
import os
import re
import logging
import config

# ==========================================
# 1. CONFIGURATION
# ==========================================
APOLLO_API_KEY = config.APOLLO_API_KEY
APOLLO_SEQUENCE_ID = "6a114515588c920014d84d3c"
SENDER_EMAIL = config.SENDER_EMAIL

CSV_FILE_PATH = "Generated_Emails_POC_Output_20260523_002947.csv"  # Ensure this matches your file

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("apollo_upload.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ==========================================
# 2. APOLLO API & HELPER FUNCTIONS
# ==========================================
def clean_email(raw_email):
    match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', str(raw_email))
    if match:
        return match.group(0).lower()
    return ""


def get_email_account_id():
    url = "https://api.apollo.io/v1/email_accounts"
    headers = {"Cache-Control": "no-cache", "X-Api-Key": APOLLO_API_KEY}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            accounts = response.json().get("email_accounts", [])
            for acc in accounts:
                if acc.get("email").lower() == SENDER_EMAIL.lower():
                    return True, acc.get("id")
            if accounts:
                logging.warning(f"Exact email '{SENDER_EMAIL}' not found. Defaulting to: {accounts[0].get('email')}")
                return True, accounts[0].get("id")
            return False, "No connected email accounts found in Apollo."

        logging.error(f"Failed to fetch mailboxes. Status: {response.status_code}, Response: {response.text}")
        return False, f"Failed to fetch mailboxes: {response.text}"
    except Exception as e:
        logging.error(f"Exception in get_email_account_id: {str(e)}", exc_info=True)
        return False, str(e)


def upsert_apollo_contact(email, first_name, last_name, company, subject, intro, fu1, fu2):
    url = "https://api.apollo.io/v1/contacts"
    headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", "X-Api-Key": APOLLO_API_KEY}

    # Passing the exact string names of the Contact Custom Fields you created
    payload = {
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
        "organization_name": company,
        "custom_fields": [
            {"name": "ai_subject", "value": subject},
            {"name": "ai_intro_body", "value": intro},
            {"name": "ai_followup_1", "value": fu1},
            {"name": "ai_followup_2", "value": fu2}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            contact_data = response.json().get("contact", {})
            return True, contact_data.get("id")
        else:
            logging.error(f"Upsert failed for {email}. Status: {response.status_code}, Response: {response.text}")
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        logging.error(f"Exception during upsert for {email}: {str(e)}", exc_info=True)
        return False, f"Request failed: {str(e)}"


def add_contact_to_sequence(contact_id, email_account_id):
    url = f"https://api.apollo.io/v1/emailer_campaigns/{APOLLO_SEQUENCE_ID}/add_contact_ids"
    headers = {"Content-Type": "application/json", "Cache-Control": "no-cache", "X-Api-Key": APOLLO_API_KEY}

    payload = {
        "contact_ids": [contact_id],
        "emailer_campaign_id": APOLLO_SEQUENCE_ID,
        "send_email_from_email_account_id": email_account_id
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return True, "Successfully added to sequence!"
        else:
            logging.error(
                f"Sequence add failed for Contact ID {contact_id}. Status: {response.status_code}, Response: {response.text}")
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        logging.error(f"Exception adding to sequence for Contact ID {contact_id}: {str(e)}", exc_info=True)
        return False, f"Request failed: {str(e)}"


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    logging.info("==================================================")
    logging.info("🚀 Starting Apollo.io Upload Sequence")
    logging.info("==================================================")

    logging.info("[*] Verifying Apollo Mailbox configuration...")
    success, email_account_id = get_email_account_id()
    if not success:
        logging.error(f"❌ Setup Error: {email_account_id}")
        return
    logging.info(f"✅ Found Mailbox ID: {email_account_id}\n")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"❌ Error: Could not find '{CSV_FILE_PATH}'.")
        return

    df_valid = df.dropna(subset=['Email ID']).copy()
    logging.info(f"Found {len(df_valid)} leads in CSV. Pushing to Apollo...\n")

    success_count = 0

    for i, (index, row) in enumerate(df_valid.iterrows(), start=1):

        raw_email = str(row.get('Email ID', '')).strip()
        clean_email_address = clean_email(raw_email)

        if not clean_email_address:
            logging.warning(
                f"[{i}/{len(df_valid)}] ⚠️ Skipping lead: Could not extract a valid email from '{raw_email}'")
            continue

        full_name = str(row.get('Reviewer Name', '')).strip()
        company = str(row.get('Reviewer Company', '')).strip()

        name_parts = full_name.split()
        first_name = name_parts[0] if name_parts else "There"
        last_name = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""

        subject = str(row.get('Intro_Subject', '')).strip()
        intro = str(row.get('Intro_Body', '')).strip()
        fu1 = str(row.get('Generated_Followup_1', '')).strip()
        fu2 = str(row.get('Generated_Followup_2', '')).strip()

        if "ERROR:" in subject or "ERROR:" in intro:
            logging.warning(f"[{i}/{len(df_valid)}] ⚠️ Skipping {clean_email_address} (AI Generation Error in CSV)")
            continue

        logging.info(f"[{i}/{len(df_valid)}] Pushing {first_name} @ {company}...")

        # Step 1: Create Contact
        success_contact, contact_result = upsert_apollo_contact(
            clean_email_address, first_name, last_name, company, subject, intro, fu1, fu2
        )

        if success_contact:
            contact_id = contact_result
            # Step 2: Add to Sequence
            seq_success, seq_result = add_contact_to_sequence(contact_id, email_account_id)

            if seq_success:
                logging.info(f"    ✅ Success: Emails uploaded & scheduled.")
                success_count += 1
            else:
                logging.error(f"    ❌ Failed to schedule: {seq_result}")
        else:
            logging.error(f"    ❌ Failed to create contact: {contact_result}")

        time.sleep(1.5)

    logging.info("==================================================")
    logging.info(f"🎉 Upload Complete! Successfully pushed {success_count} leads to Apollo.")
    logging.info("==================================================")


if __name__ == "__main__":
    main()