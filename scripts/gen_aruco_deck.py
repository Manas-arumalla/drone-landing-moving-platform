"""Generate the multi-scale ArUco deck texture (grid board + nested centre marker).

The grid keeps the platform detectable from altitude; the small centre marker, nested in the grid's
centre gap, stays in the camera FOV down to touchdown. Geometry comes from
``drone_landing.perception.board.BoardSpec``. Re-run after changing the spec.

Usage:
    $env:PYTHONPATH="src"; python scripts/gen_aruco_deck.py
"""

from __future__ import annotations

import cv2
import numpy as np

from drone_landing.perception import BoardSpec, build_board
from drone_landing.sim import repo_root

TEX_PX = 1024


def main() -> None:
    spec = BoardSpec()
    board = build_board(spec)

    board_px = int(round(spec.span / spec.deck_size * TEX_PX))
    margin = (TEX_PX - board_px) // 2
    img = board.generateImage((TEX_PX, TEX_PX), marginSize=margin, borderBits=1)

    # nest the small centre marker (different dictionary) in the grid's centre gap
    center_dict = cv2.aruco.getPredefinedDictionary(spec.center_dict_id)
    cm_px = int(round(spec.center_length / spec.deck_size * TEX_PX))
    cm = cv2.aruco.generateImageMarker(center_dict, spec.center_id, cm_px)
    off = (TEX_PX - cm_px) // 2
    # clear a white quiet zone then stamp the marker
    pad = int(0.012 * TEX_PX)
    img[off - pad:off + cm_px + pad, off - pad:off + cm_px + pad] = 255
    img[off:off + cm_px, off:off + cm_px] = cm

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = repo_root() / "assets" / "mujoco" / "textures" / "aruco_deck.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), rgb)
    print(f"wrote {out}")
    print(f"grid: {spec.markers_x}x{spec.markers_y} DICT_4X4 marker {spec.marker_length} m "
          f"sep {spec.marker_separation} m span {spec.span:.3f} m")
    print(f"centre: DICT_5X5 id {spec.center_id} size {spec.center_length} m")


if __name__ == "__main__":
    main()
