"""Vision-based perception: ArUco fiducial pose + markerless fallback.

Per the Realism Charter, perception consumes only rendered camera images (plus known intrinsics and
the onboard attitude estimate) to produce the platform's relative pose — never simulator truth."""

from drone_landing.perception.camera import CameraModel
from drone_landing.perception.board import BoardSpec, build_board
from drone_landing.perception.aruco_detector import ArucoConfig, ArucoDetection, ArucoDetector

__all__ = ["CameraModel", "BoardSpec", "build_board",
           "ArucoConfig", "ArucoDetection", "ArucoDetector"]
