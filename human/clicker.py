"""Minimal OpenCV click collector used by scripts/calib_colors.py and scripts/click_layout.py.

Keys: left-click = place the current point, u = undo, Enter/Space = accept (when complete), s = skip, q/Esc = quit.
"""
import cv2
import numpy as np

BAR = 60
COLORS_BGR = {"red": (40, 40, 230), "green": (60, 180, 60), "blue": (220, 90, 40),
              "yellow": (30, 220, 240), "purple": (170, 50, 140), "orange": (0, 130, 255)}


def collect_clicks(img, names, title="click", labels=None, hint=""):
    """Ask for one click per name, in order. Returns ({name: (x, y) image px}, status) where status is
    "ok", "skip" or "quit"."""
    labels = labels or {}
    pts = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < len(names) and y >= BAR:
            pts.append((x, y - BAR))  # store in image coordinates (the window shows a BAR-px header on top)

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    status = "quit"
    while True:
        view = img.copy()
        for (x, y), n in zip(pts, names):
            cv2.circle(view, (x, y), 7, COLORS_BGR.get(n, (255, 255, 255)), -1)
            cv2.circle(view, (x, y), 8, (0, 0, 0), 2)
            cv2.putText(view, n, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(view, n, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if len(pts) < len(names):
            n = names[len(pts)]
            msg = f"click {len(pts) + 1}/{len(names)}: {n}" + (f" ({labels[n]})" if labels.get(n, n) != n else "")
        else:
            msg = "done: Enter = accept, u = undo"
        bar = np.full((BAR, view.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar, hint + "  [u undo | s skip | q quit]", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (190, 190, 190), 1)
        cv2.imshow(title, np.concatenate([bar, view], 0))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10, 32) and len(pts) == len(names):
            status = "ok"
            break
        elif k == ord("s"):
            status = "skip"
            break
        elif k in (ord("q"), 27):
            break
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return {n: p for p, n in zip(pts, names)}, status
