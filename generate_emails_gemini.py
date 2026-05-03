#!/usr/bin/env python3
"""
Generate intro + 1st follow-up + 2nd follow-up emails with Google Gemini,
using prospect rows from the Clutch CSV and the copy rules in email_prompts.py
(sourced from Email - Prompts.pdf).

Requires:
  pip install -r requirements.txt   # includes google-generativeai
  export GEMINI_API_KEY="..."

Optional:
  export GEMINI_MODEL="gemini-2.0-flash"   # or gemini-1.5-flash, etc.

Also writes a slim CSV next to the main output (see --export-emails): LinkedIn URL,
Name, Clutch Link, Intro Mail, First Followup, Second Followup — only when generation succeeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from email_prompts import (
    FIRST_FOLLOWUP_INSTRUCTIONS,
    INTRO_EMAIL_INSTRUCTIONS,
    SECOND_FOLLOWUP_INSTRUCTIONS,
)

try:
    import google.generativeai as genai
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pip install google-generativeai\n" + str(e)
    ) from e


JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def parse_json_response(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model response")
    raw = text.strip()
    m = JSON_FENCE.match(raw)
    if m:
        raw = m.group(1).strip()
    return json.loads(raw)


def truncate_field(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 15].rstrip() + "\n…[truncated]"


def row_email(row: dict) -> str:
    for key in row:
        if key and key.strip().lower() == "email id":
            return (row.get(key) or "").strip()
    return (row.get("Email ID") or row.get("Email") or "").strip()


def get_column_ci(row: dict, *candidates: str) -> str:
    """Return cell for first candidate header that exists on the row (case-insensitive)."""
    lower_to_actual = {(k or "").strip().lower(): k for k in row}
    for cand in candidates:
        key = lower_to_actual.get(cand.strip().lower())
        if key is not None:
            return (row.get(key) or "").strip()
    return ""


EXPORT_FIELDNAMES = [
    "LinkedIn URL",
    "Name",
    "Clutch Link",
    "Intro Mail",
    "First Followup",
    "Second Followup",
]


def build_export_row(source_row: dict, rec: dict) -> dict[str, str]:
    return {
        "LinkedIn URL": get_column_ci(source_row, "LinkedIn URL", "linkedin url"),
        "Name": get_column_ci(source_row, "Reviewer Name", "name"),
        "Clutch Link": get_column_ci(source_row, "CLUTCH LINK", "clutch link"),
        "Intro Mail": (rec.get("Introductory mail") or "").strip(),
        "First Followup": (rec.get("1st Followup") or "").strip(),
        "Second Followup": (rec.get("2nd Followup") or "").strip(),
    }


def prospect_block(row: dict) -> str:
    """Compact JSON of row for the model (long fields truncated)."""
    out: dict[str, str] = {}
    for k, v in row.items():
        if v is None:
            continue
        val = str(v).strip()
        if not val:
            continue
        key = (k or "").strip()
        lk = key.lower()
        if lk in ("review text", "linkedin company about", "reach-out message"):
            val = truncate_field(val, 6000)
        out[key] = val
    return json.dumps(out, indent=2, ensure_ascii=False)


def configure_genai() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in the environment.")
    genai.configure(api_key=api_key)
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()


def generate_content(model_name: str, system: str, user: str, retries: int = 4) -> str:
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
    )
    model = genai.GenerativeModel(
        model_name,
        generation_config=generation_config,
    )
    full = (
        system.strip()
        + "\n\nYou must reply with a single valid JSON object only "
        "(no markdown fences, no commentary).\n\n--- Prospect ---\n\n"
        + user.strip()
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = model.generate_content(full)
            text = (resp.text or "").strip()
            if not text and resp.candidates:
                parts = []
                for c in resp.candidates:
                    for p in getattr(c, "content", {}).get("parts", []) if hasattr(c, "content") else []:
                        if hasattr(p, "text") and p.text:
                            parts.append(p.text)
                text = "\n".join(parts).strip()
            if not text:
                raise ValueError("Gemini returned no text (check safety filters / prompt).")
            return text
        except Exception as e:  # pragma: no cover - network
            last_err = e
            time.sleep(2**attempt)
    assert last_err is not None
    raise last_err


def gen_intro(model_name: str, prospect: str) -> tuple[str, str, str]:
    system = INTRO_EMAIL_INSTRUCTIONS + """

Return JSON with exactly these keys:
  "subject_line": string, the email subject only,
  "email_body": string, the full email body (all paragraphs and sign-off), no subject line.

Follow the client's rules above; stay within their stated character limits if humanly possible
without dropping required elements."""
    raw = generate_content(model_name, system, prospect)
    data = parse_json_response(raw)
    sub = str(data.get("subject_line", "")).strip()
    body = str(data.get("email_body", "")).strip()
    intro_mail = f"Subject: {sub}\n\n{body}" if sub else body
    return sub, body, intro_mail


def gen_followup_1(model_name: str, prospect: str, intro_mail: str) -> str:
    system = FIRST_FOLLOWUP_INSTRUCTIONS + """

Return JSON with exactly one key:
  "first_followup": string, the complete follow-up email (including sign-off).

Do not repeat praise from the intro. The intro you already sent is provided for context only."""
    user = prospect + "\n\n--- Introductory mail already sent ---\n\n" + intro_mail
    raw = generate_content(model_name, system, user)
    data = parse_json_response(raw)
    return str(data.get("first_followup", "")).strip()


def gen_followup_2(
    model_name: str, prospect: str, intro_mail: str, first_followup: str
) -> str:
    system = SECOND_FOLLOWUP_INSTRUCTIONS + """

Return JSON with exactly one key:
  "second_followup": string, the complete second follow-up (including greeting, body, and closing).

Continue the thread: do not repeat sentences from earlier mails; build on them."""
    user = (
        prospect
        + "\n\n--- Introductory mail ---\n\n"
        + intro_mail
        + "\n\n--- First follow-up ---\n\n"
        + first_followup
    )
    raw = generate_content(model_name, system, user)
    data = parse_json_response(raw)
    return str(data.get("second_followup", "")).strip()


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fields, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate intro + 2 follow-ups with Gemini from Clutch CSV + PDF prompts."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("Email - 3rd May - 27_4.csv"),
        help="Source CSV",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: gemini_emails_<input_stem>.csv)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows with email (0=all)")
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between prospects (rate limiting)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If output exists, skip rows that already have Introductory mail filled",
    )
    ap.add_argument(
        "--export-emails",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write a slim CSV (LinkedIn, Name, Clutch, 3 emails). "
        "Default: <output_stem>_emails_export.csv next to main output",
    )
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.output
    if out is None:
        out = Path(f"gemini_emails_{inp.stem}.csv")
    else:
        out = Path(out)
    out = out.resolve()

    export_emails = args.export_emails
    if export_emails is None:
        export_emails = out.with_name(out.stem + "_emails_export.csv")
    else:
        export_emails = Path(export_emails)
    export_emails = export_emails.resolve()

    model_name = configure_genai()
    orig_fields, rows = load_csv(inp)

    extra = [
        "Subject",
        "Introductory mail",
        "1st Followup",
        "2nd Followup",
        "Email",
        "gemini_model",
        "generation_error",
    ]
    out_fields = orig_fields + [c for c in extra if c not in orig_fields]

    existing_done: set[str] = set()
    if args.resume and out.is_file():
        with out.open(newline="", encoding="utf-8-sig", errors="replace") as f:
            for r in csv.DictReader(f):
                em = row_email(r)
                intro = (r.get("Introductory mail") or "").strip()
                err = (r.get("generation_error") or "").strip()
                if em and intro and not err:
                    existing_done.add(em.lower())

    processed = 0

    if not args.resume:
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore").writeheader()
    elif not out.is_file() or out.stat().st_size == 0:
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore").writeheader()

    if not args.resume:
        with export_emails.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=EXPORT_FIELDNAMES, extrasaction="ignore").writeheader()
    elif not export_emails.is_file() or export_emails.stat().st_size == 0:
        with export_emails.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=EXPORT_FIELDNAMES, extrasaction="ignore").writeheader()

    for row in rows:
        email = row_email(row)
        if not email or not EMAIL_RE.match(email):
            continue
        if args.resume and email.lower() in existing_done:
            continue

        prospect = prospect_block(row)
        rec = {**row}
        rec["Email"] = email
        rec["gemini_model"] = model_name
        rec["generation_error"] = ""

        try:
            sub, _body, intro_mail = gen_intro(model_name, prospect)
            rec["Subject"] = sub
            rec["Introductory mail"] = intro_mail
            time.sleep(max(0.0, args.sleep))
            rec["1st Followup"] = gen_followup_1(model_name, prospect, intro_mail)
            time.sleep(max(0.0, args.sleep))
            rec["2nd Followup"] = gen_followup_2(
                model_name, prospect, intro_mail, rec["1st Followup"]
            )
        except Exception as e:
            rec["generation_error"] = str(e)
            rec.setdefault("Subject", "")
            rec.setdefault("Introductory mail", "")
            rec.setdefault("1st Followup", "")
            rec.setdefault("2nd Followup", "")

        with out.open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore").writerow(rec)

        if not (rec.get("generation_error") or "").strip():
            lean = build_export_row(row, rec)
            with export_emails.open("a", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=EXPORT_FIELDNAMES, extrasaction="ignore").writerow(
                    lean
                )

        processed += 1

        if args.limit and processed >= args.limit:
            break

        time.sleep(max(0.0, args.sleep))

    print(f"Done. Wrote {processed} prospect(s) to {out} (model={model_name}).")
    print(f"Slim export (LinkedIn, Name, Clutch, 3 emails): {export_emails}")


if __name__ == "__main__":
    main()
