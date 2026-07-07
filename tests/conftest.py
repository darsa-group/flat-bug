# tests/conftest.py
import os
import subprocess


def pytest_sessionfinish(session, exitstatus):
    """This hook runs after all tests have completed."""  # noqa: D401, D404
    script_path = os.path.join(os.path.dirname(__file__), "restore_assets.py")
    print(f"\n\nRunning Post-Test Cleanup:\n\t{script_path}\n")

    try:
        subprocess.run(["python3", script_path], check=False)
    except Exception as e:
        print(f"Post-test script failed to execute: {e}")
