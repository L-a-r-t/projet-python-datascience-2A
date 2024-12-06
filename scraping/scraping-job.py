from config import fs, ROOT, ENDPOINT_URL
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timezone
import json

load_dotenv()

API_KEY = os.environ.get("IDFM_API_KEY") # Penser à mettre sa propre clé API IDF Mobilités dans .env

res = requests.get(ENDPOINT_URL, headers={"Accept": "application/json", "apiKey": API_KEY})

current_datetime = datetime.now(timezone.utc)
path = f"{ROOT}/{current_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")}.json"

with fs.open(path, "w") as f:
  json.dump(res.json(), f)