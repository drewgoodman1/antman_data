import os
import requests
from dotenv import load_dotenv

# Load API credentials from .env
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

BASE_URL = "https://data.alpaca.markets/v1beta1/options/chain"

headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "accept": "application/json",
}

params = {"underlying_symbols": "AAPL"}

print(f"Requesting: {BASE_URL}")
print(f"Headers: {headers}")
print(f"Params: {params}")

response = requests.get(BASE_URL, headers=headers, params=params)

print(f"Status code: {response.status_code}")
print(f"Response text: {response.text}")

if response.status_code == 200:
    data = response.json()
    print("Sample option chain data for AAPL:")
    for contract in data.get("option_chain", [])[:3]:
        print(contract)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
