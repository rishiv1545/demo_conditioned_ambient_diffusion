"""Generate printable ArUco markers (DICT_4X4_50, IDs 0-3, 6 cm) as assets/markers.pdf."""
import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

A4_CM = (21.0, 29.7)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/markers.pdf")
    p.add_argument("--size_cm", type=float, default=6.0)
    args = p.parse_args()
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with PdfPages(args.out) as pdf:
        for page in range(2):
            fig = plt.figure(figsize=(A4_CM[0] / 2.54, A4_CM[1] / 2.54))
            for k in range(2):
                mid = page * 2 + k
                img = cv2.aruco.generateImageMarker(d, mid, 600, borderBits=1)
                s = args.size_cm
                left = (A4_CM[0] - s) / 2
                bottom = A4_CM[1] * (0.62 if k == 0 else 0.15)
                ax = fig.add_axes([left / A4_CM[0], bottom / A4_CM[1], s / A4_CM[0], s / A4_CM[1]])
                ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
                ax.set_axis_off()
                fig.text(0.5, (bottom - 1.2) / A4_CM[1], f"ArUco DICT_4X4_50  ID {mid}  —  {s:.0f} cm side (black square incl. border)",
                         ha="center", fontsize=10)
            fig.text(0.5, 0.97, "Print at 100% / actual size (no 'fit to page'). Verify the 6 cm side with a ruler.",
                     ha="center", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
