import os
import subprocess
import sys
import time


def test_tool_server_module_reports_a_port(tmp_path):
    port_file = tmp_path / "port"
    env = {
        **os.environ,
        "MCP_PORT_FILE": str(port_file),
        "VF_CONFIG": "{}",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "symbolic_tool_calling_v1.servers.tools"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for _ in range(50):
            if port_file.exists():
                break
            if process.poll() is not None:
                raise AssertionError(process.stderr.read())
            time.sleep(0.1)
        assert port_file.read_text().strip().isdigit()
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)
