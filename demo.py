import subprocess
import xgboost as xgb
import pandas as pd
import time
import os

# === CONFIGURATION ===
MODEL_PATH = "model.json"
FEATURES = [
    "cache-references", "cache-misses", "cycles", "instructions",
    "branches", "branch-misses", "L1-dcache-loads", "L1-dcache-load-misses",
    "LLC-loads", "LLC-load-misses", "dTLB-loads", "dTLB-load-misses"
]
SAMPLE_INTERVAL = 5  # Seconds between each perf capture

def run_perf():
    cmd = [
        "perf", "stat", "-a", "-x", ",",
        "-e", ",".join(FEATURES),
        "sleep", str(SAMPLE_INTERVAL)
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    return result.stderr

def parse_perf_output(perf_output):
    lines = perf_output.strip().splitlines()
    data = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        try:
            val = int(parts[0].replace(",", "").strip())
        except ValueError:
            val = 0
        event = parts[2].strip()
        if event in FEATURES:
            data[event] = val
    return [data.get(f, 0) for f in FEATURES]

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

def monitor():
    print("Loading model...")
    model = load_model()
    print("Monitoring started. Press Ctrl+C to stop.\n")

    while True:
        print(f"[{time.strftime('%H:%M:%S')}] Collecting perf data...")
        perf_output = run_perf()
        features = parse_perf_output(perf_output)
        df = pd.DataFrame([features], columns=FEATURES)

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]  # Confidence of "attack" class

        if prediction == 1:
            print(f"ATTACK DETECTED! Confidence: {prob:.2f}")
        else:
            print(f"BENIGN activity. Confidence: {prob:.2f}")

        print("-" * 50)
        time.sleep(1)  # Optional cooldown between iterations

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
