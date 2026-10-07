"""Conservative AC-power gate for explicitly expensive recovery batches."""
import subprocess
import sys

def on_ac_power():
    if sys.platform != 'darwin':
        return True
    result=subprocess.run(['pmset','-g','batt'],capture_output=True,text=True,timeout=5)
    return result.returncode==0 and "Now drawing from 'AC Power'" in result.stdout
