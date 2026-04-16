"""
generate_sample.py
──────────────────
Creates a synthetic classroom scene PNG  (sample_classroom.png)
with simple stick-figure silhouettes arranged in rows.

Run once to produce a test image without needing a real photo:
    python generate_sample.py
"""

import cv2
import numpy as np


def draw_person(img, cx, cy, scale=1.0, color=(60, 100, 180)):
    """Draw a simple filled silhouette at (cx, cy)."""
    s = scale
    # body
    cv2.rectangle(img,
                  (int(cx - 12*s), int(cy)),
                  (int(cx + 12*s), int(cy + 35*s)),
                  color, -1)
    # head
    cv2.circle(img, (int(cx), int(cy - 12*s)), int(12*s), color, -1)
    # left leg
    cv2.rectangle(img,
                  (int(cx - 11*s), int(cy + 35*s)),
                  (int(cx - 2*s),  int(cy + 60*s)),
                  color, -1)
    # right leg
    cv2.rectangle(img,
                  (int(cx + 2*s),  int(cy + 35*s)),
                  (int(cx + 11*s), int(cy + 60*s)),
                  color, -1)


def create_classroom_image(path="sample_classroom.png"):
    # Canvas: warm off-white classroom wall
    h, w = 480, 720
    img  = np.full((h, w, 3), (220, 210, 195), dtype=np.uint8)

    # Draw a simple "desk row" background
    for row_y in [340, 390, 440]:
        cv2.rectangle(img, (0, row_y), (w, row_y + 8), (150, 130, 110), -1)

    # Place 9 students in 3 rows
    positions = []
    for row, (base_y, s) in enumerate([(250, 1.0), (200, 0.88), (155, 0.76)]):
        count = 3
        xs    = [int(w * f) for f in [0.18, 0.5, 0.82]]
        for cx in xs:
            shade = max(40, 80 - row * 15)
            color = (shade + 30, shade + 60, shade + 130)
            draw_person(img, cx, base_y, scale=s, color=color)
            positions.append((cx, base_y))

    # Blackboard at top
    cv2.rectangle(img, (60, 20), (660, 110), (40, 80, 60), -1)
    cv2.putText(img, "CLASSROOM  –  9 students",
                (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 230, 200), 2)

    cv2.imwrite(path, img)
    print(f"[OK] Synthetic classroom image saved → {path}")
    print(f"     Size: {w}×{h}, students drawn: {len(positions)}")


if __name__ == "__main__":
    create_classroom_image()
