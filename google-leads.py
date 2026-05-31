import requests
import json
import os
import time
from datetime import datetime
import config

# ==========================================
# CONFIGURATION
# ==========================================
SERPAPI_KEY = config.SERPAPI_KEY

# 🚀 EXPANDED SOFTWARE CONSULTANCY QUERIES
SEARCH_QUERIES = [
    # --- CATEGORY 1: Public RFPs & Project Requirements ---
    'site:notion.site "Request for Proposal" OR "RFP" "software development" OR "app development"',
    '"Request for Proposal" OR "RFP" "machine learning" OR "AI" OR "computer vision" filetype:pdf',
    '"project requirements" AND "MVP development" OR "custom software" -jobs',
    '"Request for Quote" OR "RFQ" "custom software" OR "web application" filetype:pdf',
    'site:docs.google.com/document "project scope" OR "requirements" "app development"',

    # --- CATEGORY 2: Contract Hiring & Staff Augmentation ---
    'site:wellfound.com/jobs "Contract" "Machine Learning" OR "React" OR "Node.js"',
    'site:weworkremotely.com OR site:remoteok.com "Contract" OR "Freelance" "Software Engineer" OR "Developer"',
    'site:news.ycombinator.com/item "Freelance" OR "Contract" OR "Agency" "hiring"',
    'site:bamboohr.com/careers "Contractor" OR "Freelance" "Software Engineer" OR "Data Scientist"',
    'site:greenhouse.io OR site:lever.co "Staff Augmentation" OR "Contract Developer"',

    # --- CATEGORY 3: Founder Forums & Communities ---
    'site:reddit.com/r/SaaS OR site:reddit.com/r/startups "looking for an agency" OR "dev shop" OR "development partner"',
    'site:reddit.com/r/cofounder "looking for a technical" OR "need a developer" "AI" OR "MVP"',
    'site:news.ycombinator.com "Ask HN: Seeking freelancer" OR "Ask HN: Who is hiring? (Contract)"',
    'site:indiehackers.com "looking for developers" OR "need an agency" OR "development partner"',

    # --- CATEGORY 4: Enterprise Tech Debt & Cloud Migration ---
    'site:boards.greenhouse.io OR site:jobs.lever.co "cloud migration" OR "technical debt" "AWS" OR "DevOps"',
    '"legacy modernization" OR "cloud migration" "RFP" OR "Request for Proposal" filetype:pdf',
    '"digital transformation" "software development partner" OR "vendor"',

    # --- CATEGORY 5: The "Funded Startup" Trigger ---
    'site:techcrunch.com OR site:venturebeat.com "Series A" OR "Series B" AND "AI" OR "SaaS" "raised"',
    'site:coindesk.com OR site:techcrunch.com "seed round" OR "Series A" AND "Web3" OR "Blockchain" "raised"',
    'site:prweb.com OR site:businesswire.com "Series A" AND "IoT" OR "Internet of Things" "funding"',

    # --- CATEGORY 6: Niche Verticals ---
    '"HIPAA compliant" "software development" "RFP" OR "Request for Proposal"',
    '"fintech" OR "banking" "MVP development" "agency" OR "vendor"',
    '"Shopify Plus" "custom app" OR "migration" "RFP" OR "project requirements"',
    '"smart contract audit" OR "dApp development" "RFP" OR "bounty"',

    # --- CATEGORY 7: Fractional Leadership triggers ---
    'site:linkedin.com/jobs "Fractional CTO" OR "Part-time CTO" "seeking" OR "hiring"',
    'site:wellfound.com/jobs "Fractional CTO" OR "Technical Advisor"'
]


def fetch_serp_leads(query, api_key):
    """Calls SerpApi to get Google Search results for a specific query."""
    print(f"\n🔍[SEARCHING] {query}")

    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": 100,  # UPDATED: Maximize results per API credit (Google max is 100)
        "gl": "us",  # Target US Market
        "hl": "en",
        "tbs": "qdr:m"  # Filters results to the PAST MONTH
    }

    try:
        response = requests.get(url, params=params, timeout=15)  # ADDED: Timeout to prevent hanging
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching data for query '{query}': {e}")
        return None


def process_results(data, seen_urls):
    """Extracts relevant lead data and ensures no duplicate URLs are added."""
    leads = []

    if "organic_results" in data:
        for result in data["organic_results"]:
            url = result.get("link")

            # DEDUPLICATION LOGIC: Skip if we've already found this exact URL
            if url in seen_urls:
                continue

            seen_urls.add(url)

            lead = {
                "title": result.get("title"),
                "url": url,
                "snippet": result.get("snippet", "").replace('\n', ' '),  # Clean up multiline snippets
                "date": result.get("date", "Recent")
            }
            leads.append(lead)
    else:
        print("   [-] No organic results found for this specific query (in the past month).")

    return leads


def main():
    if not SERPAPI_KEY:
        print(
            "\n[!] ACTION REQUIRED: SERPAPI_KEY is missing. Add it to your .env file (see .env.example).\n")
        return

    all_leads = []
    seen_urls = set()  # ADDED: Set to track and prevent duplicate leads

    print("=========================================================")
    print(" 🚀 STARTING SOFTWAREBRIO LEAD GENERATION ENGINE")
    print("=========================================================\n")

    for query in SEARCH_QUERIES:
        data = fetch_serp_leads(query, SERPAPI_KEY)

        if data:
            extracted_leads = process_results(data, seen_urls)

            if extracted_leads:
                print(f"   [+] Found {len(extracted_leads)} UNIQUE leads:")
                print("   " + "-" * 50)

                # PRINTING TO CONSOLE
                for lead in extracted_leads[:3]:  # Only print first 3 to prevent terminal flood
                    print(f"   🔹 TITLE:   {lead['title']}")
                    print(f"   🔗 URL:     {lead['url']}")
                    print(f"   📝 SNIPPET: {lead['snippet']}")
                    print(f"   📅 DATE:    {lead['date']}")
                    print("   " + "-" * 50)

                if len(extracted_leads) > 3:
                    print(f"   ... and {len(extracted_leads) - 3} more. (Check JSON for all)")

            all_leads.extend(extracted_leads)

        # RATE LIMITING: Sleep for 1.5 seconds between requests
        time.sleep(1.5)

    # Save everything to a JSON file at the end
    filename = f"softwarebrio_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_leads, f, indent=4, ensure_ascii=False)

    print("\n=========================================================")
    print(f" 🎉 SUCCESS! {len(all_leads)} TOTAL UNIQUE LEADS FOUND")
    print(f" 💾 All leads have been saved to: {filename}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()