import urllib.request
import json
import time

base_url = "http://127.0.0.1:8000/api/v1"

def send_post(endpoint, payload):
    url = f"{base_url}/{endpoint}"
    req_body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_body,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8.0) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            print(f"[{endpoint}] SUCCESS")
            print(json.dumps(res_json, indent=2))
            return res_json
    except Exception as e:
        print(f"[{endpoint}] FAILED: {e}")
        return None

print("=== 1. Testing M3 Multimodal PEFT/LoRA Inference ===")
multimodal_res = send_post("predict/multimodal", {
    "description": "Severe public protest blocking main corridor near Mekhri Circle",
    "comment": "barricades set up by police",
    "event_cause": "protest",
    "corridor": "Bellary Road"
})

print("\n=== 2. Testing M4 Graph WaveNet GNN Traffic Inference ===")
traffic_res1 = send_post("predict/traffic", {
    "lat": [13.0084],
    "lng": [77.5906],
    "reported_datetime": "2026-06-20T18:00:00+05:30"
})

print("\n=== 3. Testing Kafka Event Streaming Ingest ===")
ingest_res = send_post("stream/ingest", {
    "event_id": "evt_9901",
    "event_cause": "protest",
    "corridor": "Bellary Road",
    "priority": "High",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "Ingested via Kafka stream",
    "comment": "Live update"
})

print("\n=== 4. Testing TomTom Real-Time API/Sensor Telemetry Ingest ===")
telemetry_res = send_post("stream/traffic", {
    "lat": 13.0084,
    "lng": 77.5906,
    "speed_kmh": 12.5,
    "flow_veh_hr": 1450,
    "congestion_index": 0.85
})

print("\n=== 5. Testing Blended GNN + Real-Time Telemetry Traffic Inference ===")
# This should show the status "+Kafka_TomTom_Live_Telemetry_Ingested" and updated values
traffic_res2 = send_post("predict/traffic", {
    "lat": [13.0084],
    "lng": [77.5906],
    "reported_datetime": "2026-06-20T18:00:00+05:30"
})

print("\n=== 6. Testing Preprocessing Robustness to Out-of-Vocabulary (OOV) Category ===")
closure_oov_res = send_post("predict/closure", {
    "event_cause": "unseen_cause_xxx",
    "corridor": "unseen_corridor_yyy",
    "priority": "unseen_priority_zzz",
    "reported_datetime": "2026-06-20T18:00:00+05:30",
    "description": "OOV testing",
    "comment": "safe test",
    "vehicle_type": "unseen_vehicle_type_www"
})
