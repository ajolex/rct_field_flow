# rct_field_flow/surveycto.py
"""
SurveyCTO API client for fetching form submissions and uploading cases.

API Documentation: https://docs.surveycto.com/
Note: The data export API requires a 'date' parameter to filter submissions.
"""
import requests
import pandas as pd
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def _normalize_server(server: str) -> str:
    """Return a clean SurveyCTO host like 'yourserver.surveycto.com'.

    Accepts inputs like 'yourserver', 'yourserver.surveycto.com', or
    'https://yourserver.surveycto.com/'. Also fixes accidental duplicates like
    'yourserver.surveycto.com.surveycto.com'.
    """
    s = (server or "").strip()
    if not s:
        return s
    if s.startswith("http://") or s.startswith("https://"):
        p = urlparse(s)
        s = p.netloc or p.path
    # drop any path/params after host
    s = s.split("/")[0]
    # remove accidental userinfo
    if "@" in s:
        s = s.split("@")[-1]
    s = s.lower()
    # collapse duplicate domain suffixes
    while s.endswith(".surveycto.com.surveycto.com"):
        s = s[:-len(".surveycto.com")]  # remove one duplicate suffix
    # ensure single correct suffix
    if not s.endswith(".surveycto.com"):
        # if just subdomain provided, append official domain
        if "." not in s:
            s = f"{s}.surveycto.com"
    return s


class SurveyCTO:
    def __init__(self, server: str, username: str, password: str):
        host = _normalize_server(server)
        self.base_url = f"https://{host}/api/v2"
        self.auth = (username, password)

    def get_submissions(self, form_id: str, since: Optional[str] = None) -> pd.DataFrame:
        """
        Get form submissions from SurveyCTO.
        
        Args:
            form_id: The form ID to fetch submissions for
            since: Optional date string in format 'Oct 15, 2024 12:00:00 AM' or timestamp.
                   If None, defaults to '0' to get all submissions.
        
        Returns:
            DataFrame containing the submissions
        """
        url = f"{self.base_url}/forms/data/wide/json/{form_id}"
        # SurveyCTO API requires the 'date' parameter. Use '0' to get all data.
        # Format can be: 'Oct 15, 2024 12:00:00 AM', timestamp, or '0' for all data
        params = {"date": since if since else "0"}
        response = requests.get(url, auth=self.auth, params=params, timeout=60)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def upload_cases(self, csv_path: str, form_id: str):
        url = f"{self.base_url}/cases/upload"
        files = {"file": open(csv_path, "rb")}
        data = {"form_id": form_id}
        response = requests.post(url, files=files, auth=self.auth, data=data, timeout=120)
        return response.json()
