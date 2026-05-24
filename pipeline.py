"""
Outreach pipeline — orchestrates ai-studio-2.py and apollo-push-v2.py.

Step 1  Generate emails          (ai-studio-2.py → generate_for_lead, research_lead_with_google)
Step 2  Validate every field     (pipeline-only logic; retry generation up to MAX_GEN_RETRIES)
Step 3  Push to Apollo           (apollo-push-v2.py → sync_to_apollo, get_active_email_account_id)

Leads that fail all retries are saved to *_FAILED.csv and skipped from Apollo.
"""

import importlib.util
import pandas as pd
import json
import time
import os
from datetime import datetime

# ============================================================
# PIPELINE CONFIGURATION
# ============================================================
INPUT_CSV        = "Email - 3rd May - 27_4.csv"
LEAD_LIMIT       = 30      # set None to process all rows
MAX_GEN_RETRIES  = 3       # retries per lead if validation fails
RATE_LIMIT_SLEEP = 17      # seconds between leads (Gemini Free Tier: 15 RPM)

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"
FAILED_CSV = f"Generated_Emails_FAILED_{timestamp}.csv"

# ============================================================
# LOAD ai-studio-2.py AND apollo-push-v2.py AS MODULES
# Files use hyphens so importlib is required instead of import.
# ============================================================
_here = os.path.dirname(os.path.abspath(__file__))

def _load(filename, module_name):
    path = os.path.join(_here, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gen  = _load("ai-studio-2.py",    "ai_studio")   # email generation
push = _load("apollo-push-v2.py", "apollo_push")  # Apollo sync

# ============================================================
# VALIDATION — pipeline-only logic
# Checks every generated field before allowing it to be pushed.
# ============================================================
FIELD_RULES = {
    #  field_key                  max_words  require_bold_tag
    "Intro_Subject":              (10,       False),
    "Intro_Body":                 (67,       True),
    "Generated_Followup_1":       (54,       True),
    "Generated_Followup_2":       (58,       True),
}


def validate_lead_emails(row):
    """
    Returns (passed: bool, issues: list[str]).
    Checks every required field for: non-empty, no ERROR marker,
    word count within limit, no banned words, required <b> tag present.
    """
    issues = []
    for field, (max_words, require_bold) in FIELD_RULES.items():
        value = str(row.get(field, "")).strip()

        if not value:
            issues.append(f"{field}: EMPTY")
            continue
        if "ERROR:" in value:
            issues.append(f"{field}: generation error — {value[:80]}")
            continue
        if "Skipped due to" in value:
            issues.append(f"{field}: skipped due to upstream error")
            continue

        word_count = len(value.split())
        if word_count > max_words:
            issues.append(f"{field}: too long ({word_count}/{max_words} words)")

        if gen.contains_banned_words(value):
            found = [w for w in gen.BANNED_WORDS if w in value.lower()]
            issues.append(f"{field}: banned word(s) {found}")

        if require_bold and "<b>" not in value:
            issues.append(f"{field}: missing required <b> HTML tag")

    return (len(issues) == 0), issues


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 65)
    print("  OUTREACH PIPELINE")
    print("  Step 1: Generate  →  Step 2: Validate  →  Step 3: Push")
    print("=" * 65)

    if not os.path.exists(INPUT_CSV):
        print(f"[!] Input CSV not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    df_filtered = df.dropna(subset=["Email ID"]).copy()
    if LEAD_LIMIT:
        df_filtered = df_filtered.head(LEAD_LIMIT)
    total = len(df_filtered)
    print(f"\n[*] Loaded {total} leads from '{INPUT_CSV}'\n")

    for col in ["Intro_Subject", "Intro_Body", "Generated_Followup_1", "Generated_Followup_2", "Live_Research_Data"]:
        df_filtered[col] = ""

    validated_rows = []
    failed_rows    = []
    validation_log = []

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

        # Research (ai-studio-2.py)
        print(f"  [*] Researching via Google Search...")
        live_research = gen.research_lead_with_google(name, company, linkedin)
        df_filtered.at[index, "Live_Research_Data"] = live_research

        lead_context = (
            f"CONTEXT FOR THIS LEAD:\n"
            f"Name: {name}\nLinkedIn: {linkedin}\nCompany: {company}\n"
            f"Past Msg: {reachout}\nProject Review: {review}\nCompany About: {about}\n\n"
            f"LIVE GOOGLE SEARCH RESEARCH:\n{live_research}"
        )

        best_result = None
        best_issues = None

        for attempt in range(1, MAX_GEN_RETRIES + 1):
            print(f"  [*] Generating emails (attempt {attempt}/{MAX_GEN_RETRIES})...")

            # Generate (ai-studio-2.py)
            generated = gen.generate_for_lead(lead_context, first_name, company)

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
                if best_issues is None or len(issues) < len(best_issues):
                    best_result = generated
                    best_issues = issues

        # Write best result back to dataframe
        for col, val in best_result.items():
            df_filtered.at[index, col] = val

        row_out = df_filtered.loc[index].to_dict()

        # Print full generated content (same as ai-studio-2.py standalone)
        print(json.dumps({
            "Lead": f"{first_name} at {company}",
            "Live_Research_Found": live_research,
            "Validation": "PASS" if not best_issues else f"FAIL — {best_issues}",
            "Messages": {
                "Subject":     best_result.get("Intro_Subject", ""),
                "Intro_Body":  best_result.get("Intro_Body", ""),
                "Follow-up_1": best_result.get("Generated_Followup_1", ""),
                "Follow-up_2": best_result.get("Generated_Followup_2", ""),
            },
        }, indent=4, ensure_ascii=False))

        if not best_issues:
            validated_rows.append(row_out)
            validation_log.append({"email": email, "status": "PASS", "issues": []})
        else:
            failed_rows.append(row_out)
            validation_log.append({"email": email, "status": "FAIL", "issues": best_issues})
            print(f"  [✗] Lead failed after {MAX_GEN_RETRIES} attempt(s) — will NOT be pushed to Apollo.")

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
        print("[!] No validated leads to push.")
        return

    # get_active_email_account_id (apollo-push-v2.py)
    email_account_id = push.get_active_email_account_id()
    if not email_account_id:
        print("[!] Could not resolve sender inbox. Aborting Apollo push.")
        return

    push_success = 0
    push_failed  = 0

    for i, lead in enumerate(validated_rows, start=1):
        email = lead.get("Email ID", "")
        print(f"\n{'─'*65}")
        print(f"  Pushing ({i}/{len(validated_rows)}): {email}")
        print(f"{'─'*65}")

        # sync_to_apollo (apollo-push-v2.py) — prints full JSON debug internally
        ok = push.sync_to_apollo(lead, email_account_id)
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
