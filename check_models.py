import os
from google import genai

# Load your key
API_KEY = "AIzaSyBiup-AaqYKz1EKDPX7v7oTepwVaL3Jduc"
client = genai.Client(api_key=API_KEY)

print("Fetching approved models for your API key...\n")

try:
    # Get the raw list of models your key has access to
    models = client.models.list()

    for m in models:
        # The new SDK might return 'models/gemini-...' or just 'gemini-...'
        # We will clean it up so it's easy to read
        clean_name = m.name.replace("models/", "")
        print(f"- {clean_name}")

except Exception as e:
    print(f"Failed to fetch models: {e}")