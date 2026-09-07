"""Stop whatever process is holding the demo server's port (default 5000).

Fixes the "OSError: [WinError 10048] Only one usage of each socket
address..." error you get from 09_server.py when a previous run (or a
stray background process) never released the port.

Usage:
    python scripts/shutdown.py           # kills whatever is on port 5000
    python scripts/shutdown.py 8080      # or any other port
"""

import subprocess
import sys


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "5000"

    result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)

    pids = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[1].endswith(f":{port}"):
            pids.add(parts[-1])

    pids.discard("0")
    if not pids:
        print(f"Nothing is listening on port {port}. Nothing to do.")
        return

    for pid in pids:
        print(f"Killing PID {pid} (was listening on port {port})...")
        subprocess.run(["taskkill", "/PID", pid, "/F"])

    print("Done.")


if __name__ == "__main__":
    main()
