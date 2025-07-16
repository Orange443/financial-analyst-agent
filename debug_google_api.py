import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

print(f"--- Running Minimal Debug Script ---")
print(f"API Key Loaded: {bool(GOOGLE_API_KEY)}")
print(f"CSE ID Loaded: {bool(GOOGLE_CSE_ID)}")

# Check if credentials are loaded
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    print("\nError: Could not load GOOGLE_API_KEY or GOOGLE_CSE_ID from .env file.")
else:
    # The base URL for the Custom Search API
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Parameters for the API request
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": "TCS.NS stock news",
        "num": 1
    }

    print("\nAttempting to make a direct API call with the `requests` library...")

    try:
        response = requests.get(url, params=params)
        
        # Check the status code
        print(f"Response Status Code: {response.status_code}")
        
        # Raise an exception for bad status codes (4xx or 5xx) to see the error
        response.raise_for_status()
        
        # If successful, print the result
        print("\n--- SUCCESS! ---")
        print("API call was successful. The issue lies within the main agent's script or its dependencies.")
        print("First result:", response.json().get('items', [{}])[0].get('title'))

    except requests.exceptions.HTTPError as e:
        print("\n--- FAILURE! ---")
        print(f"An HTTP error occurred: {e}")
        print("\nResponse Body:")
        print(e.response.text)
        print("\nThis confirms the issue is with the Python environment's ability to connect to the Google API, not the script's logic.")
    except Exception as e:
        print("\n--- FAILURE! ---")
        print(f"An unexpected error occurred: {e}")
