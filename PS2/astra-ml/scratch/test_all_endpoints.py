import urllib.request
import json
import socket

base_url = "http://127.0.0.1:8000/api/v1/predict"

def send_post(endpoint, payload):
    url = f"{base_url}/{endpoint}"
    req_body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            print(f"[{endpoint}] SUCCESS:")
            print(json.dumps(res_json, indent=2))
            return res_json
    except Exception as e:
        print(f"[{endpoint}] FAILED: {e}")
        return None

print("=== Testing Closure Endpoint ===")
closure_res = send_post("closure", {
    "event_cause": "accident",
    "corridor": "Mysore Road",
    "priority": "High",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "Accident on Mysore Road",
    "comment": "blocked",
    "vehicle_type": "heavy_vehicle",
    "junction": "K R Circle",
    "zone": "Central Zone 1"
})

print("\n=== Testing Duration Endpoint (Acute) ===")
duration_acute_res = send_post("duration", {
    "event_cause": "accident",
    "corridor": "Mysore Road",
    "priority": "High",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "Accident on Mysore Road",
    "comment": "blocked",
    "vehicle_type": "heavy_vehicle",
    "junction": "K R Circle",
    "zone": "Central Zone 1"
})

print("\n=== Testing Duration Endpoint (Chronic) ===")
duration_chronic_res = send_post("duration", {
    "event_cause": "construction",
    "corridor": "Mysore Road",
    "priority": "Medium",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "Metro construction works",
    "comment": "slow traffic, barricades placed",
    "vehicle_type": None,
    "junction": "K R Circle",
    "zone": "Central Zone 1"
})

print("\n=== Testing Multimodal Endpoint ===")
multimodal_res = send_post("multimodal", {
    "description": "Protest rally demanding local transport benefits",
    "comment": "highly unstable crowd",
    "event_cause": "protest",
    "corridor": "Mysore Road"
})

print("\n=== Testing Traffic Endpoint ===")
traffic_res = send_post("traffic", {
    "junction": "MekhriCircle",
    "reported_datetime": "2026-06-20T18:00:00+05:30"
})
