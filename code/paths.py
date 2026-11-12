"""Folders of the replication package: data/ (inputs and solutions), output/ (exhibits), and logs/.
Imported by the scripts in code/; creates output/ and logs/ if missing."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
LOGS = os.path.join(ROOT, "logs")
for _d in (OUT, LOGS):
    os.makedirs(_d, exist_ok=True)
