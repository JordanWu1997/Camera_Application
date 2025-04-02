#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
[ADD MODULE DOCUMENTATION HERE]

# ========================================================================== #
#  _  __   _   _                                          __        ___   _  #
# | |/ /  | | | |  Author: Jordan Kuan-Hsien Wu           \ \      / / | | | #
# | ' /   | |_| |  E-mail: jordankhwu@gmail.com            \ \ /\ / /| | | | #
# | . \   |  _  |  Github: https://github.com/JordanWu1997  \ V  V / | |_| | #
# |_|\_\  |_| |_|  Datetime: 2024-12-22 16:11:23             \_/\_/   \___/  #
#                                                                            #
# ========================================================================== #
"""

import argparse
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import parse_video_device


def get_available_devices(number_of_devices=10, max_index=1000, verbose=False):
    """  """
    index, found_devices = 0, 0
    devices = []
    while (found_devices <= number_of_devices) and (index < max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            devices.append(index)
            found_devices += 1
        cap.release()
        index += 1
    if verbose:
        print(devices)
    return devices


class DetectionSystem:

    def __init__(self):
        # Initialize MediaPipe components
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Initialize detectors
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.5)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # Custom drawing specs
        self.face_drawing_spec = self.mp_draw.DrawingSpec(color=(0, 255, 0),
                                                          thickness=1,
                                                          circle_radius=1)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for hand gestures, face mesh, and pose detection.

        Args:
            frame: Input frame from video stream

        Returns:
            Tuple containing processed frame and detection results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process all detections
        hand_results = self.hands.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        # Create annotation layer
        annotation_frame = frame.copy()

        # Process and draw detections
        results = self._process_detections(
            annotation_frame, hand_results.multi_hand_landmarks,
            face_results.multi_face_landmarks
            if face_results.multi_face_landmarks else [],
            pose_results.pose_landmarks)

        # Blend annotation layer with original frame
        alpha = 0.7
        frame = cv2.addWeighted(annotation_frame, alpha, frame, 1 - alpha, 0)

        return frame, results

    def _process_detections(self, frame: np.ndarray, hand_landmarks,
                            face_landmarks, pose_landmarks) -> Dict:
        """
        Process and visualize all detections.

        Args:
            frame: Input frame
            hand_landmarks: MediaPipe hand landmarks
            face_landmarks: MediaPipe face landmarks
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            Dictionary containing detection results
        """
        height, width, _ = frame.shape
        results = {
            "gestures": [],
            "face_landmarks": False,
            "pose_detected": False
        }

        # Process hand landmarks
        if hand_landmarks:
            for landmarks in hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2))

                # Recognize gesture
                gesture = self._recognize_gesture(landmarks)
                results["gestures"].append(gesture)

                # Add gesture text
                x = int(landmarks.landmark[0].x * width)
                y = int(landmarks.landmark[0].y * height)
                cv2.putText(frame, gesture, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)

        # Process face landmarks
        if face_landmarks:
            for face_landmark in face_landmarks:
                # Draw face mesh
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw_styles.
                    get_default_face_mesh_tesselation_style())

                # Draw face contours
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw_styles.
                    get_default_face_mesh_contours_style())

                # Draw irises
                self.mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_draw_styles.
                    get_default_face_mesh_iris_connections_style())

                results["face_landmarks"] = True

        # Process pose landmarks
        if pose_landmarks:
            # Draw pose landmarks
            self.mp_draw.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw_styles.
                get_default_pose_landmarks_style())

            results["pose_detected"] = True

            # Calculate and display angle for some key joints (e.g., elbows)
            if pose_landmarks:
                results["pose_angles"] = self._calculate_pose_angles(
                    pose_landmarks)

        return results

    def _recognize_gesture(self, landmarks) -> str:
        """
        Recognize specific gestures based on hand landmark positions.
        """
        # Get fingertip landmarks
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]

        # Check if fingers are extended
        fingers_extended = [
            thumb_tip.x < landmarks.landmark[3].x,  # Thumb
            index_tip.y < landmarks.landmark[6].y,  # Index
            middle_tip.y < landmarks.landmark[10].y,  # Middle
            ring_tip.y < landmarks.landmark[14].y,  # Ring
            pinky_tip.y < landmarks.landmark[18].y  # Pinky
        ]

        # Recognize gestures
        if all(fingers_extended):
            return "Open Hand"
        elif not any(fingers_extended):
            return "Fist"
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            return "Pointing"
        elif fingers_extended[1] and fingers_extended[2] and not any(
                fingers_extended[3:]):
            return "Peace"
        else:
            return "Unknown"

    def _calculate_pose_angles(self, landmarks) -> Dict[str, float]:
        """
        Calculate angles for key pose joints.
        """

        def calculate_angle(a, b, c) -> float:
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])

            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                     np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180.0:
                angle = 360 - angle

            return angle

        # Calculate key joint angles
        angles = {
            "left_elbow":
            calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]),
            "right_elbow":
            calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        }

        return angles


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_device',
                        default=None,
                        type=str,
                        help='Input device, file or strearming URL')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    args = parser.parse_args()

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Initialize video capture
    cap = cv2.VideoCapture(input_device)

    # Check if stream opened successfully
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Initialize detection system
    detector = DetectionSystem()

    # Process frames
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from stream.")
                break

            # Process frame
            start_time = time.time()
            processed_frame, results = detector.process_frame(frame)

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the processed frame
            cv2.imshow('Comprehensive Detection System', processed_frame)

            # Print detection results
            print("Detections:", results)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {str(e)}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
