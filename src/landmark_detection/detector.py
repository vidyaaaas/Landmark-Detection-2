from __future__ import annotations

from dataclasses import dataclass

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
    """Detect and draw face, hand, and pose landmarks on images."""

    def __init__(self, options: DetectorOptions | None = None) -> None:
        self.options = options or DetectorOptions()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

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

    def close(self) -> None:
        self.face_mesh.close()
        self.hands.close()
        self.pose.close()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a copy of the frame with detected landmarks drawn."""
        output = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        if self.options.detect_faces:
            self._draw_faces(frame_rgb, output)
        if self.options.detect_hands:
            self._draw_hands(frame_rgb, output)
        if self.options.detect_pose:
            self._draw_pose(frame_rgb, output)

        return output

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
