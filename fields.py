import requests
import config

APOLLO_API_KEY = config.APOLLO_API_KEY


def fetch_custom_fields():
    url = "https://api.apollo.io/v1/contact_custom_fields"
    headers = {
        "Cache-Control": "no-cache",
        "X-Api-Key": APOLLO_API_KEY
    }

    print("Fetching Custom Field IDs from Apollo...\n")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            fields = response.json().get("contact_custom_fields", [])
            for f in fields:
                name = f.get("name")
                field_id = f.get("id")
                print(f"Name: {name} | ID: {field_id}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to fetch fields: {str(e)}")


if __name__ == "__main__":
    fetch_custom_fields()