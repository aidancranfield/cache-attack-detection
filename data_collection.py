import subprocess
import csv
import time
import os
import signal
import random
from itertools import product

# === CONFIGURATION ===
TOTAL_SAMPLES = 288 * 8   # where N = samples per attack combo
SAMPLE_DURATION = 5
COOLDOWN_TIME = 2
OUTPUT_CSV = "test.csv"

ATTACKS = {
    "fr": "./attacks/fr/fr",
    "pp": "./attacks/pp/pp",
    "ff": "./attacks/ff/ff",
    "er": "./attackd/er/er"
}

TARGET_OFFSETS = {
    "/lib/x86_64-linux-gnu/libc.so.6": "0x60100",
    "/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1": "0xbba40",
    "/usr/lib/x86_64-linux-gnu/libssl.so.1.1": "0x22550",
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6": "0xbd000",
    "/lib/x86_64-linux-gnu/libpam.so.0": "0x4c40",
    "/usr/lib/x86_64-linux-gnu/libz.so.1": "0x1810",
    "/usr/lib/x86_64-linux-gnu/libX11.so.6": "0x4080",
    "/lib/x86_64-linux-gnu/libm.so.6": "0x1d000",
    "/lib/x86_64-linux-gnu/libsystemd.so.0": "0x2000",
    "/usr/lib/x86_64-linux-gnu/libsqlite3.so.0": "0x1500",
    "/usr/lib/x86_64-linux-gnu/libgcrypt.so.20": "0x7000",
    "/lib64/ld-linux-x86-64.so.2": "0x10e0"
}
TARGET_LIBS = list(TARGET_OFFSETS.keys())

LOAD_MODES = {
    "no-load": None,
    "avg-load": ["stress-ng", "--cpu", "2", "--timeout", "6s", "--quiet"],
    "full-load": ["stress-ng", "--cpu", "4", "--vm", "2", "--vm-bytes", "512M", "--timeout", "6s", "--quiet"]
}

BENIGN_COMMANDS = [
    ["/usr/bin/ls", "-lR", "/usr"],
    ["/bin/cat", "/etc/passwd"],
    ["/usr/bin/grep", "root", "/etc/passwd"],
    ["/usr/bin/openssl", "rand", "-hex", "64"],
    ["/usr/bin/curl", "-s", "https://example.com"],
    ["/usr/bin/apt", "list"],
    ["/usr/bin/g++", "--version"],
    ["/usr/bin/sudo", "-V"],
    ["/usr/bin/loginctl"],
    ["/bin/tar", "cf", "/dev/null", "/etc"],
    ["/usr/bin/gzip", "--version"],
    ["/usr/bin/xprop"],
    ["/usr/bin/python3", "-c", "import math; print(math.exp(1))"],
    ["/usr/bin/systemctl", "list-units"],
    ["/usr/bin/sqlite3", ":memory:", "SELECT 1;"],
    ["/usr/bin/gpg", "--version"]
]

PERF_EVENTS = (
    "cache-references,cache-misses,cycles,instructions,branches,branch-misses,"
    "L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,"
    "dTLB-loads,dTLB-load-misses"
)

def run_perf_stat(label, attack_type, load_label, bg_commands, sample_id):
    perf_cmd = [
        "timeout", "--signal=SIGINT", str(SAMPLE_DURATION),
        "perf", "stat", "-x,", "-a", "-e", PERF_EVENTS
    ]

    bg_procs = []
    load_proc = None
    try:
        # Start load if needed
        if load_label != "no-load":
            load_proc = subprocess.Popen(LOAD_MODES[load_label])
            print(f"[LOAD] {load_label} started (PID: {load_proc.pid})")

        # Start multiple background commands
        for cmd in bg_commands:
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            bg_procs.append(p)
            print(f"[BG] {cmd} (PID: {p.pid})")

        # Run system-wide perf stat
        perf_result = subprocess.run(
            perf_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=SAMPLE_DURATION + 2
        )

        # Cleanup
        for p in bg_procs:
            if p.poll() is None:
                os.kill(p.pid, signal.SIGKILL)
                p.wait()
        if load_proc and load_proc.poll() is None:
            os.kill(load_proc.pid, signal.SIGKILL)
            load_proc.wait()

        # Parse perf output
        lines = perf_result.stderr.decode().strip().split("\n")
        values = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    val = int(parts[0].replace(",", ""))
                    values.append(val)
                except:
                    pass

        if len(values) == 12:
            return [sample_id] + values + [label, attack_type, load_label]
        else:
            print(f"[!] Incomplete perf data for {sample_id}: {values}")

    except Exception as e:
        print(f"[!] Error in sample {sample_id}: {e}")
        for p in bg_procs:
            if p.poll() is None:
                os.kill(p.pid, signal.SIGKILL)
                p.wait()
        if load_proc and load_proc.poll() is None:
            os.kill(load_proc.pid, signal.SIGKILL)
            load_proc.wait()

    return None

def main():
    attack_samples = TOTAL_SAMPLES // 2
    benign_samples = TOTAL_SAMPLES // 2

    attack_combos = list(product(ATTACKS.items(), TARGET_LIBS, LOAD_MODES.keys()))
    samples_per_attack_combo = attack_samples // len(attack_combos)
    benign_samples_per_load = benign_samples // len(LOAD_MODES)

    print(f"Attack samples =\t\t {attack_samples}")
    print(f"Benign samples =\t\t {benign_samples}")
    print(f"Attack combots =\t\t {attack_combos}")
    print(f"Samples per attack combo  =\t\t {samples_per_attack_combo}")
    print(f"Samples per benign combo =\t\t {benign_samples_per_load}")


    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "cache-references", "cache-misses", "cycles", "instructions",
            "branches", "branch-misses", "L1-dcache-loads", "L1-dcache-load-misses",
            "LLC-loads", "LLC-load-misses", "dTLB-loads", "dTLB-load-misses",
            "label", "attack_type", "load_level"
        ])

        # ATTACK SAMPLES
        for ((attack_type, attack_path), target_lib, load_label) in attack_combos:
            offset = TARGET_OFFSETS[target_lib]
            for i in range(samples_per_attack_combo):
                sample_id = f"{attack_type}_{os.path.basename(target_lib)}_{load_label}_{i}"
                cmd = [[attack_path, target_lib, offset]]
                row = run_perf_stat(1, attack_type, load_label, cmd, sample_id)
                if row:
                    writer.writerow(row)
                time.sleep(COOLDOWN_TIME)

        # BENIGN SAMPLES
        for load_label in LOAD_MODES:
            for i in range(benign_samples_per_load):
                sample_id = f"benign_{load_label}_{i}"
                num_procs = random.randint(2, 4)
                cmds = random.sample(BENIGN_COMMANDS, num_procs)
                row = run_perf_stat(0, "benign", load_label, cmds, sample_id)
                if row:
                    writer.writerow(row)
                time.sleep(COOLDOWN_TIME)

    print(f"Dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
