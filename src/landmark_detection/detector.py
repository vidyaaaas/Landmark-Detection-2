from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class DetectorOptions:
    detect_faces: bool = True
    detect_hands: bool = True
    detect_pose: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class LandmarkDetector:
    """Detect and draw landmarks using MediaPipe when available, with an OpenCV fallback."""

    def __init__(self, options: DetectorOptions | None = None) -> None:
        self.options = options or DetectorOptions()
        self.backend = "opencv"
        self.face_mesh = None
        self.hands = None
        self.pose = None

        if self._setup_mediapipe_solutions():
            self.backend = "mediapipe"
            return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )

    def _setup_mediapipe_solutions(self) -> bool:
        solutions = getattr(mp, "solutions", None)
        if solutions is not None:
            self.mp_drawing = solutions.drawing_utils
            self.mp_styles = solutions.drawing_styles
            self.mp_face_mesh = solutions.face_mesh
            self.mp_hands = solutions.hands
            self.mp_pose = solutions.pose
        else:
            try:
                self.mp_drawing = import_module("mediapipe.solutions.drawing_utils")
                self.mp_styles = import_module("mediapipe.solutions.drawing_styles")
                self.mp_face_mesh = import_module("mediapipe.solutions.face_mesh")
                self.mp_hands = import_module("mediapipe.solutions.hands")
                self.mp_pose = import_module("mediapipe.solutions.pose")
            except ModuleNotFoundError:
                return False

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=self.options.min_detection_confidence,
            min_tracking_confidence=self.options.min_tracking_confidence,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.options.min_detection_confidence,
            min_tracking_confidence=self.options.min_tracking_confidence,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.options.min_detection_confidence,
            min_tracking_confidence=self.options.min_tracking_confidence,
        )
        return True

    def close(self) -> None:
        for detector in (self.face_mesh, self.hands, self.pose):
            if detector is not None:
                detector.close()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a copy of the frame with detected landmarks drawn."""
        if self.backend == "mediapipe":
            return self._process_with_mediapipe(frame_bgr)
        return self._process_with_opencv(frame_bgr)

    def _process_with_mediapipe(self, frame_bgr: np.ndarray) -> np.ndarray:
        output = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self.options.detect_faces:
            self._draw_faces(frame_rgb, output)
        if self.options.detect_hands:
            self._draw_hands(frame_rgb, output)
        if self.options.detect_pose:
            self._draw_pose(frame_rgb, output)

        cv2.putText(
            output,
            "Backend: MediaPipe",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 220, 40),
            2,
        )
        return output

    def _process_with_opencv(self, frame_bgr: np.ndarray) -> np.ndarray:
        output = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for x, y, w, h in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 220, 255), 2)

            roi_gray = gray[y : y + h, x : x + w]
            roi_output = output[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(20, 20))
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.7, 20, minSize=(25, 12))

            self._draw_face_reference_points(output, x, y, w, h)
            for ex, ey, ew, eh in eyes[:2]:
                center = (x + ex + ew // 2, y + ey + eh // 2)
                cv2.circle(output, center, max(3, ew // 8), (0, 255, 0), 2)
                cv2.circle(output, center, 3, (0, 255, 0), -1)

            for sx, sy, sw, sh in smiles[:1]:
                cv2.rectangle(roi_output, (sx, sy), (sx + sw, sy + sh), (255, 120, 0), 2)

        cv2.putText(
            output,
            "Backend: OpenCV fallback",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )
        return output

    @staticmethod
    def _draw_face_reference_points(output: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        points = [
            (x + w // 2, y + h // 5),
            (x + w // 3, y + h // 3),
            (x + (2 * w) // 3, y + h // 3),
            (x + w // 2, y + h // 2),
            (x + w // 3, y + (2 * h) // 3),
            (x + (2 * w) // 3, y + (2 * h) // 3),
            (x + w // 2, y + (4 * h) // 5),
        ]
        for point in points:
            cv2.circle(output, point, 3, (255, 0, 255), -1)

    def _draw_faces(self, frame_rgb: np.ndarray, output: np.ndarray) -> None:
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return

        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style(),
            )
            self.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style(),
            )

    def _draw_hands(self, frame_rgb: np.ndarray, output: np.ndarray) -> None:
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

    def _draw_pose(self, frame_rgb: np.ndarray, output: np.ndarray) -> None:
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return

        self.mp_drawing.draw_landmarks(
            output,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
        )
