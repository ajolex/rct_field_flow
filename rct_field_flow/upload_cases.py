from __future__ import annotations

import requests
from .surveycto import _normalize_server


def upload_to_surveycto(
    csv_path: str,
    server: str,
    username: str,
    password: str,
    form_id: str | None = None,
    mode: str = "merge",
) -> dict:
    """
    Upload a case roster to SurveyCTO case management via the HTTP API.
    
    Args:
        csv_path: Path to CSV file with cases
        server: SurveyCTO server URL
        username: SurveyCTO username
        password: SurveyCTO password
        form_id: Form ID to upload cases for
        mode: Upload mode - 'merge' (default), 'append', or 'replace'
            - merge: Update existing cases and add new ones
            - append: Only add new cases, skip existing ones
            - replace: Delete all existing cases and upload only these
    
    Returns:
        API response dictionary
    """
    if mode not in ["merge", "append", "replace"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'merge', 'append', or 'replace'")
    
    host = _normalize_server(server)
    
    # Use the case management endpoint with form_id in URL
    if not form_id:
        raise ValueError("form_id is required for case uploads")
    
    url = f"https://{host}/api/v2/forms/data/caselist/{form_id}"
    data = {"mode": mode}
    auth = (username, password)
    
    with open(csv_path, "rb") as handle:
        files = {"file": handle}
        response = requests.post(url, files=files, auth=auth, data=data, timeout=120)
    
    response.raise_for_status()
    return response.json()
