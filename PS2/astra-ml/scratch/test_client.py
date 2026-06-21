import urllib.request
import json
import socket

url = "http://127.0.0.1:8000/api/v1/predict/closure"
data = {
    "event_cause": "accident",
    "corridor": "Mysore Road",
    "priority": "High",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "Accident on Mysore Road",
    "comment": "blocked"
}
req_body = json.dumps(data).encode("utf-8")

req = urllib.request.Request(
    url,
    data=req_body,
    headers={"Content-Type": "application/json"}
)

print("Sending request to FastAPI predict closure endpoint...")
try:
    with urllib.request.urlopen(req, timeout=5.0) as response:
        html = response.read().decode("utf-8")
        print("Success! Response:")
        print(html)
except socket.timeout:
    print("Error: socket timeout! Connection timed out.")
except urllib.error.URLError as e:
    print(f"Error: URLError: {e.reason}")
except Exception as e:
    print(f"Error: {e}")
