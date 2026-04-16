"""
test_setup.py – Validate that all dependencies are installed correctly.

Run:  python test_setup.py
"""

import sys


def check(name: str, import_stmt: str) -> bool:
    try:
        exec(import_stmt)
        print(f"  ✔  {name}")
        return True
    except Exception as e:
        print(f"  ✘  {name}  –  {e}")
        return False


print("\n── Checking dependencies ────────────────────────────")
results = [
    check("numpy",         "import numpy as np; print(f'       numpy {np.__version__}', end='')"),
    check("opencv-python", "import cv2;          print(f'       cv2   {cv2.__version__}', end='')"),
    check("ultralytics",   "import ultralytics;  print(f'       ult   {ultralytics.__version__}', end='')"),
    check("torch",         "import torch;        print(f'       torch {torch.__version__}', end='')"),
    check("streamlit",     "import streamlit;    print(f'       st    {streamlit.__version__}', end='')"),
]
print()

if all(results):
    print("── All dependencies OK ✔  ────────────────────────────")
    print("\nTrying to load YOLOv8n (will download on first run) …")
    try:
        from ultralytics import YOLO
        m = YOLO("yolov8n.pt")
        print(f"  ✔  Model loaded  –  {len(m.names)} classes")
        print(f"  ✔  'person' class id = "
              f"{[k for k,v in m.names.items() if v=='person'][0]}")
    except Exception as e:
        print(f"  ✘  Model load failed: {e}")
else:
    missing = sum(1 for r in results if not r)
    print(f"── {missing} missing package(s). Run:  pip install -r requirements.txt")
    sys.exit(1)

print("\nSetup check complete. You are ready to run the project!\n")
