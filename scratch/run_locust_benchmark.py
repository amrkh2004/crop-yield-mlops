import os
import subprocess
import sys
import time
import urllib.request


def main():
    print("Starting uvicorn API server...")
    venv_python = sys.executable
    server_process = subprocess.Popen(
        [venv_python, "-m", "uvicorn", "prodml.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server readiness
    ready = False
    for attempt in range(10):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
                if resp.status == 200:
                    ready = True
                    print("API server is healthy!")
                    break
        except Exception:
            pass

    if not ready:
        print("Failed to start API server.")
        server_process.terminate()
        sys.exit(1)

    os.makedirs("reports", exist_ok=True)
    locust_exe = os.path.join(os.path.dirname(venv_python), "locust.exe")
    if not os.path.exists(locust_exe):
        locust_exe = "locust"

    print("Running Locust headless load test...")
    cmd = [
        locust_exe,
        "-f",
        "locustfile.py",
        "--headless",
        "-u",
        "20",
        "-r",
        "5",
        "--run-time",
        "15s",
        "--host",
        "http://127.0.0.1:8000",
        "--html",
        "reports/locust_summary.html",
        "--csv",
        "reports/locust",
    ]

    locust_res = subprocess.run(cmd, capture_output=True, text=True)
    print("Locust STDOUT:\n", locust_res.stdout)
    print("Locust STDERR:\n", locust_res.stderr)

    print("Terminating uvicorn server...")
    server_process.terminate()
    server_process.wait()


if __name__ == "__main__":
    main()
