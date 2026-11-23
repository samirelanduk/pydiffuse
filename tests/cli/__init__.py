import subprocess
import sys

IN_MAC_VM = (
    sys.platform == "darwin"
    and subprocess.run(
        ["sysctl", "-n", "kern.hv_vmm_present"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    == "1"
)
