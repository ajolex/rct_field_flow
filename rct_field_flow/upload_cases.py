from __future__ import annotations

import requests


def upload_to_surveycto(
    csv_path: str,
    server: str,
    username: str,
    password: str,
    form_id: str | None = None,
) -> dict:
    """Upload a case roster to SurveyCTO case management via the HTTP API."""
    url = f"https://{server}.surveycto.com/api/v2/cases/upload"
    data = {"form_id": form_id} if form_id else {}
    auth = (username, password)
    with open(csv_path, "rb") as handle:
        files = {"file": handle}
        response = requests.post(url, files=files, auth=auth, data=data, timeout=120)
    response.raise_for_status()
    return response.json()
