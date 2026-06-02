"""
Outreach pipeline — orchestrates ai-studio-2.py and apollo-push-v2.py.

Step 1  Generate emails          (ai-studio-2.py → generate_for_lead, research_lead_with_google)
Step 2  Validate every field     (pipeline-only logic; retry generation up to MAX_GEN_RETRIES)
Step 3  Push to Apollo           (apollo-push-v2.py → sync_to_apollo, get_active_email_account_id)

Leads that fail all retries are saved to *_FAILED.csv and skipped from Apollo.
All output is mirrored to a .log file with timestamps so nothing is lost after terminal scroll.
"""

import importlib.util
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime

# ============================================================
# PIPELINE CONFIGURATION
# ============================================================
INPUT_CSV        = "CLUTCH_LEAD_GEN - 4_5.csv"
LEAD_LIMIT       = 10                 # set None to process all rows
MAX_GEN_RETRIES  = 3       # retries per lead if validation fails
RATE_LIMIT_SLEEP = 30      # seconds between leads — gives the 5 RPM (Gemini 2.5 Pro free) more breathing room

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"Generated_Emails_POC_Output_{timestamp}.csv"
FAILED_CSV = f"Generated_Emails_FAILED_{timestamp}.csv"
LOG_FILE   = f"pipeline_run_{timestamp}.log"


# ============================================================
# LOGGING — mirrors all print() output to a .log file so
# nothing is lost after terminal scroll or crash.
# ============================================================
class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _ts():
    """Return current time as [HH:MM:SS] prefix for log lines."""
    return datetime.now().strftime("[%H:%M:%S]")

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
    # max_chars = the target written into the prompts (informational only).
    # Char count is NO LONGER a validation failure — only banned words,
    # missing <b> tags, empty values, and API errors cause retries.
    #  field_key                  max_chars  require_bold_tag
    "Intro_Subject":              (65,       False),
    "Intro_Body":                 (330,      True),
    "Generated_Followup_1":       (340,      True),
    "Generated_Followup_2":       (370,      True),
}



def validate_lead_emails(row, print_counts=False):
    """
    Returns (passed: bool, issues: list[str]).
    Checks every required field for: non-empty, no ERROR marker,
    char count within limit, no banned words, required <b> tag present.

    Char count is measured on the email body ONLY — the signature appended
    by generate_for_lead() is stripped before counting so it doesn't eat
    into the email's character budget.

    When print_counts=True prints a per-field char-count table so you can
    see how close each field is to the limit.
    """
    issues = []
    sig_suffix = "\n\n" + gen.SIGNATURE  # signature appended programmatically
    counts = []

    for field, (max_chars, require_bold) in FIELD_RULES.items():
        value = str(row.get(field, "")).strip()

        if not value:
            issues.append(f"{field}: EMPTY")
            counts.append((field, 0, max_chars, "EMPTY"))
            continue
        if "ERROR:" in value:
            issues.append(f"{field}: generation error — {value[:80]}")
            counts.append((field, 0, max_chars, "ERROR"))
            continue
        if "Skipped due to" in value:
            issues.append(f"{field}: skipped due to upstream error")
            counts.append((field, 0, max_chars, "SKIPPED"))
            continue

        # Strip signature before char count — it's ~50 chars that Gemini
        # never wrote and should not count against the limit.
        body_only = value.replace(sig_suffix, "").strip()

        char_count = len(body_only)
        counts.append((field, char_count, max_chars, "ok"))

        if gen.contains_banned_words(body_only):
            found = [w for w in gen.BANNED_WORDS if w in body_only.lower()]
            issues.append(
                f"{field}: banned word(s) {found} — "
                f"content: «{body_only[:120]}{'…' if len(body_only) > 120 else ''}»"
            )

        if require_bold and "<b>" not in value:
            issues.append(
                f"{field}: missing required <b> HTML tag — "
                f"content: «{body_only[:120]}{'…' if len(body_only) > 120 else ''}»"
            )

    if print_counts:
        print(f"  {'Field':<28} {'Chars':>6} {'Target':>7}")
        print(f"  {'─'*28} {'─'*6} {'─'*7}")
        for field, cc, lim, status in counts:
            if status in ("EMPTY", "ERROR", "SKIPPED"):
                print(f"  {field:<28} {'—':>6} {lim:>7}  ⚠ {status}")
            else:
                margin = lim - cc
                bar = "✅" if margin >= 0 else "~"
                print(f"  {field:<28} {cc:>6} {lim:>7} {f'+{margin}' if margin >= 0 else str(margin):>8}  {bar}")

    return (len(issues) == 0), issues


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    # Mirror all stdout to a persistent log file for this run
    _log_handle = open(LOG_FILE, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_handle)

    print("=" * 65)
    print(f"  OUTREACH PIPELINE   run={timestamp}")
    print("  Step 1: Generate  →  Step 2: Validate  →  Step 3: Push")
    print(f"  Log file: {LOG_FILE}")
    print("=" * 65)

    if not os.path.exists(INPUT_CSV):
        print(f"[!] Input CSV not found: {INPUT_CSV}")
        sys.stdout = sys.__stdout__
        _log_handle.close()
        return

    df = pd.read_csv(INPUT_CSV)
    df_filtered = df.dropna(subset=["Email ID"]).copy()
    if LEAD_LIMIT:
        df_filtered = df_filtered.head(LEAD_LIMIT)
    total = len(df_filtered)
    print(f"\n{_ts()} [*] Loaded {total} leads from '{INPUT_CSV}'\n")

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
        print(f"  {_ts()} Lead {i}/{total}: {first_name} at {company} <{email}>")
        print(f"{'─'*65}")

        # Research (ai-studio-2.py)
        print(f"  {_ts()} [*] Researching via Google Search...")
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
        quota_exhausted = False  # set True when all fields come back as ERRORs

        for attempt in range(1, MAX_GEN_RETRIES + 1):
            print(f"  {_ts()} [*] Generating emails (attempt {attempt}/{MAX_GEN_RETRIES})...")

            # Generate (ai-studio-2.py)
            generated = gen.generate_for_lead(lead_context, first_name, company)

            # Detect sustained quota exhaustion: if EVERY generated field is an
            # error (not a validation miss), the API is completely unavailable.
            # Retrying immediately wastes quota — pause before the next attempt.
            all_errors = all(
                str(generated.get(f, "")).startswith("ERROR:")
                or str(generated.get(f, "")).startswith("Skipped")
                for f in ["Intro_Subject", "Intro_Body", "Generated_Followup_1", "Generated_Followup_2"]
            )
            if all_errors and attempt < MAX_GEN_RETRIES:
                quota_exhausted = True
                print(f"  {_ts()} [!] All fields returned errors — likely quota exhausted. Waiting 120s before retry...")
                time.sleep(120)

            candidate = {**row.to_dict(), **generated}
            passed, issues = validate_lead_emails(candidate, print_counts=True)

            if passed:
                best_result = generated
                best_issues = []
                print(f"  {_ts()} [+] Validation PASSED (attempt {attempt}).")
                break
            else:
                print(f"  {_ts()} [!] Validation FAILED (attempt {attempt}/{MAX_GEN_RETRIES}):")
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
            row_out["Failure_Reasons"] = " | ".join(best_issues)
            failed_rows.append(row_out)
            validation_log.append({"email": email, "status": "FAIL", "issues": best_issues})
            print(f"  {_ts()} [✗] Lead failed after {MAX_GEN_RETRIES} attempt(s) — will NOT be pushed to Apollo.")

        if i < total:
            print(f"\n  {_ts()} Waiting {RATE_LIMIT_SLEEP}s for rate limits...")
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
    print(f"  {_ts()} STEP 3: Pushing {len(validated_rows)} validated lead(s) to Apollo")
    print(f"{'='*65}\n")

    if not validated_rows:
        print("[!] No validated leads to push.")
        sys.stdout = sys.__stdout__
        _log_handle.close()
        return

    # get_active_email_account_id (apollo-push-v2.py)
    email_account_id = push.get_active_email_account_id()
    if not email_account_id:
        print("[!] Could not resolve sender inbox. Aborting Apollo push.")
        sys.stdout = sys.__stdout__
        _log_handle.close()
        return

    push_success = 0
    push_failed  = 0

    for i, lead in enumerate(validated_rows, start=1):
        email = lead.get("Email ID", "")
        print(f"\n{'─'*65}")
        print(f"  {_ts()} Pushing ({i}/{len(validated_rows)}): {email}")
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
    print(f"  PIPELINE COMPLETE   {_ts()}")
    print(f"{'='*65}")
    print(f"  Total leads processed : {total}")
    print(f"  Generation passed     : {len(validated_rows)}")
    print(f"  Generation failed     : {len(failed_rows)}")
    print(f"  Apollo push succeeded : {push_success}")
    print(f"  Apollo push failed    : {push_failed}")

    if failed_rows:
        print(f"\n  ── Leads that need manual review ({FAILED_CSV}) ──")
        for entry in validation_log:
            if entry["status"] == "FAIL":
                print(f"    • {entry['email']}")
                for issue in entry["issues"]:
                    # issue already contains a content snippet when applicable
                    print(f"        ↳ {issue}")

    print(f"\n  Output CSV : {OUTPUT_CSV}")
    if failed_rows:
        print(f"  Failed CSV : {FAILED_CSV}  (includes 'Failure_Reasons' column)")
    print(f"  Log file   : {LOG_FILE}")
    print()

    sys.stdout = sys.__stdout__
    _log_handle.close()
    print(f"[✓] Log written to {LOG_FILE}")


if __name__ == "__main__":
    main()
