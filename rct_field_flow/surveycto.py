# rct_field_flow/surveycto.py
import requests
import pandas as pd
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class SurveyCTO:
    def __init__(self, server: str, username: str, password: str):
        self.base_url = f"https://{server}.surveycto.com/api/v2"
        self.auth = (username, password)

    def get_submissions(self, form_id: str, since: Optional[str] = None) -> pd.DataFrame:
        url = f"{self.base_url}/forms/data/wide/json/{form_id}"
        params = {"date": since} if since else {}
        response = requests.get(url, auth=self.auth, params=params)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def upload_cases(self, csv_path: str, form_id: str):
        url = f"{self.base_url}/cases/upload"
        files = {"file": open(csv_path, "rb")}
        data = {"form_id": form_id}
        response = requests.post(url, files=files, auth=self.auth, data=data)
        return response.json()