#!/usr/bin/env python3
"""
Build plain-text outreach emails from a Clutch export CSV (e.g. Email - 3rd May - 27_4.csv).

Reads rows with a non-empty Email ID, produces Subject + Body columns for mail merge or paste.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
CONTENT_REF_RE = re.compile(r":contentReference\[[^\]]*\]\{[^}]*\}")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = CONTENT_REF_RE.sub("", s)
    return " ".join(s.split()).strip()


def first_name(reviewer_name: str) -> str:
    n = (reviewer_name or "").strip()
    if not n:
        return "there"
    return n.split()[0]


def strip_trailing_signoff(body: str) -> str:
    """Remove the LinkedIn-style closing bundled into Reach-out Message."""
    t = body.strip()
    t = re.sub(
        r"\s*Looking forward to connecting\.\s*Parag\s+Google\s*\|\s*Founder\s+SoftwareBrio\.com\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\s*Looking forward to connecting\.\s*$", "", t, flags=re.IGNORECASE
    )
    t = re.sub(
        r"\s*Parag\s+Google\s*\|\s*Founder\s+SoftwareBrio\.com\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    )
    return t.strip()


def format_email_body(reach_out: str, clutch_url: str) -> str:
    core = clean_text(strip_trailing_signoff(reach_out))
    if not core:
        return ""

    # Break after "Hi Name,"
    core = re.sub(
        r"^(Hi\s+[^,]+,)\s+",
        r"\1\n\n",
        core,
        count=1,
        flags=re.IGNORECASE,
    )

    parts = [core]

    if clutch_url.strip():
        parts.append("")
        parts.append(
            "P.S. Your Clutch review is what prompted this note — sharing the link for context:\n"
            f"{clutch_url.strip()}"
        )

    parts.append("")
    parts.append("Looking forward to connecting.")
    parts.append("")
    parts.append("Best,")
    parts.append("Parag")
    parts.append("Founder, SoftwareBrio.com")
    parts.append("Ex-Google")

    return "\n".join(parts)


def subject_line(row: dict) -> str:
    fn = first_name(row.get("Reviewer Name", ""))
    vendor = (row.get("Outsourced Company Name") or "").strip()
    company = (row.get("Reviewer Company") or "").strip()

    if vendor:
        cand = f"{fn}, quick note on your work with {vendor}"
        if len(cand) <= 78:
            return cand
        cand = f"{fn}, your project with {vendor}"
        if len(cand) <= 78:
            return cand

    if company:
        cand = f"{fn}, quick note — {company}"
        if len(cand) <= 78:
            return cand

    return f"{fn}, quick question"


def row_email(row: dict) -> str:
    for key in row:
        if key.strip().lower() == "email id":
            return (row.get(key) or "").strip()
    return (row.get("Email ID") or row.get("Email") or "").strip()


def load_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def main() -> None:
    p = argparse.ArgumentParser(description="Generate outreach email drafts from Clutch CSV.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Email - 3rd May - 27_4.csv"),
        help="Source CSV path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: outreach_emails_<input_stem>.csv)",
    )
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.output
    if out is None:
        out = Path(f"outreach_emails_{inp.stem}.csv")
    else:
        out = Path(out)

    _, rows = load_rows(inp)

    out_fieldnames = [
        "Email ID",
        "Reviewer Name",
        "Reviewer Company",
        "Outsourced Company Name",
        "Service Outsourced",
        "Subject",
        "Body",
        "CLUTCH LINK",
    ]

    written = 0
    skipped_no_email = 0
    skipped_invalid = 0

    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        w.writeheader()

        for row in rows:
            email = row_email(row)
            if not email:
                skipped_no_email += 1
                continue
            if not EMAIL_RE.match(email):
                skipped_invalid += 1
                continue

            reach = row.get("Reach-out Message") or ""
            clutch = row.get("CLUTCH LINK") or ""

            rec = {
                "Email ID": email,
                "Reviewer Name": (row.get("Reviewer Name") or "").strip(),
                "Reviewer Company": (row.get("Reviewer Company") or "").strip(),
                "Outsourced Company Name": (row.get("Outsourced Company Name") or "").strip(),
                "Service Outsourced": (row.get("Service Outsourced") or "").strip(),
                "Subject": subject_line(row),
                "Body": format_email_body(reach, clutch),
                "CLUTCH LINK": clutch.strip(),
            }
            w.writerow(rec)
            written += 1

    print(f"Wrote {written} rows to {out}")
    print(f"Skipped (no email): {skipped_no_email}, skipped (invalid format): {skipped_invalid}")


if __name__ == "__main__":
    main()
