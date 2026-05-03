# test_api.py
import requests
import json
import numpy as np

BASE = "http://localhost:5000"

# Generate a realistic synthetic PPG (4 seconds at 125 Hz = 500 samples)
def make_ppg():
    fs  = 125
    t   = np.linspace(0, 4, 4 * fs)
    hr  = 70 / 60.0
    ppg = 0.6 * np.sin(2 * np.pi * hr * t)
    ppg += 0.2 * np.sin(4 * np.pi * hr * t - 0.3)
    ppg += 0.1 * np.sin(6 * np.pi * hr * t - 0.6)
    ppg += 0.02 * np.random.randn(len(t))
    ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
    return ppg.tolist()

PPG = make_ppg()   # 500 samples, realistic pulse shape

def test(label, method, url, **kwargs):
    print(f"\n── {label}")
    r = getattr(requests, method)(url, **kwargs)
    print(f"   Status : {r.status_code}")
    print(f"   Response: {json.dumps(r.json(), indent=4)}")

# Test 1 — server alive
test("Status check", "get", f"{BASE}/status")

# Test 2 — valid PPG prediction
test("Valid prediction", "post", f"{BASE}/predict",
     json={"ppg": PPG})

# Test 3 — too short (should return 400)
test("Too short PPG", "post", f"{BASE}/predict",
     json={"ppg": [0.5, 0.6, 0.7]})

# Test 4 — missing field (should return 400)
test("Missing ppg field", "post", f"{BASE}/predict",
     json={})

# Test 5 — history (should now have 1 entry from Test 2)
test("History", "get", f"{BASE}/history?n=5")