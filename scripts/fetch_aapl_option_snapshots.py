import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

url = "https://data.alpaca.markets/v1beta1/options/snapshots/AAPL"
headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "accept": "application/json",
}
params = {"feed": "indicative", "limit": 5}  # try indicative (delayed) feed

response = requests.get(url, headers=headers, params=params)
print("Status code:", response.status_code)
print("Response:", response.text)
