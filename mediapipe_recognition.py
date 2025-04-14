#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
MediaPipe-based detection system with gesture recognition, face mesh tracking, and pose estimation.
Features:
- Frame skipping for performance optimization
- Toggleable detection modes with keyboard shortcuts
- Zoom in/out and pan controls for the video stream
- FPS display and performance monitoring

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
import math
import time
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import parse_video_device, resize_image


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

        # Finger landmark indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20

        # Finger MCP (base) indices
        self.THUMB_MCP = 2
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.RING_MCP = 13
        self.PINKY_MCP = 17

        # Wrist landmark
        self.WRIST = 0

        # Feature toggle flags (default all on)
        self.enable_hand_detection = True
        self.enable_face_detection = True
        self.enable_pose_detection = True

        # Skip frame counter
        self.frame_count = 0
        self.skip_frames = 0  # Process every frame by default

        # Cache for detection results when skipping frames
        self.last_hand_landmarks = None
        self.last_face_landmarks = None
        self.last_pose_landmarks = None

        # Zoom and pan parameters
        self.zoom_factor = 1.0
        self.pan_x = 0.0  # Pan offset as percentage of frame width (-0.5 to 0.5)
        self.pan_y = 0.0  # Pan offset as percentage of frame height (-0.5 to 0.5)
        self.zoom_step = 0.1
        self.pan_step = 0.05

    def set_skip_frames(self, skip_frames):
        """Set number of frames to skip between detections"""
        self.skip_frames = max(0, skip_frames)  # Ensure non-negative

    def toggle_hand_detection(self):
        """Toggle hand detection on/off"""
        self.enable_hand_detection = not self.enable_hand_detection
        return self.enable_hand_detection

    def toggle_face_detection(self):
        """Toggle face detection on/off"""
        self.enable_face_detection = not self.enable_face_detection
        return self.enable_face_detection

    def toggle_pose_detection(self):
        """Toggle pose detection on/off"""
        self.enable_pose_detection = not self.enable_pose_detection
        return self.enable_pose_detection

    def zoom_in(self):
        """Increase zoom level"""
        self.zoom_factor += self.zoom_step
        return self.zoom_factor

    def zoom_out(self):
        """Decrease zoom level (minimum zoom is 1.0)"""
        self.zoom_factor = max(1.0, self.zoom_factor - self.zoom_step)
        return self.zoom_factor

    def pan_left(self):
        """Pan view left"""
        # Limit pan range based on zoom level to prevent viewing outside the frame
        max_pan = (self.zoom_factor - 1.0) / (2.0 * self.zoom_factor)
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x - self.pan_step))
        return self.pan_x

    def pan_right(self):
        """Pan view right"""
        max_pan = (self.zoom_factor - 1.0) / (2.0 * self.zoom_factor)
        self.pan_x = max(-max_pan, min(max_pan, self.pan_x + self.pan_step))
        return self.pan_x

    def pan_up(self):
        """Pan view up"""
        max_pan = (self.zoom_factor - 1.0) / (2.0 * self.zoom_factor)
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y - self.pan_step))
        return self.pan_y

    def pan_down(self):
        """Pan view down"""
        max_pan = (self.zoom_factor - 1.0) / (2.0 * self.zoom_factor)
        self.pan_y = max(-max_pan, min(max_pan, self.pan_y + self.pan_step))
        return self.pan_y

    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        return (self.zoom_factor, self.pan_x, self.pan_y)

    def apply_zoom_pan(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom and pan transformations to frame"""
        if self.zoom_factor == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
            return frame  # No transformation needed

        h, w = frame.shape[:2]

        # Calculate region of interest
        zoom_w = int(w / self.zoom_factor)
        zoom_h = int(h / self.zoom_factor)

        # Calculate center offset with pan
        center_x = int(w / 2 + self.pan_x * w)
        center_y = int(h / 2 + self.pan_y * h)

        # Calculate ROI boundaries
        x1 = max(0, int(center_x - zoom_w / 2))
        y1 = max(0, int(center_y - zoom_h / 2))
        x2 = min(w, int(center_x + zoom_w / 2))
        y2 = min(h, int(center_y + zoom_h / 2))

        # Extract ROI
        zoomed_frame = frame[y1:y2, x1:x2]

        # Resize back to original dimensions
        return cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame for hand gestures, face mesh, and pose detection.

        Args:
            frame: Input frame from video stream

        Returns:
            Tuple containing processed frame and detection results
        """
        # Apply zoom and pan before processing
        processed_view = self.apply_zoom_pan(frame)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(processed_view, cv2.COLOR_BGR2RGB)

        # Check if we should process this frame or use cached results
        should_process = (self.frame_count % (self.skip_frames + 1) == 0)
        self.frame_count += 1

        # Process detections based on enabled features and skip frame logic
        hand_landmarks = None
        face_landmarks = []
        pose_landmarks = None

        if should_process:
            # Only run detection if the feature is enabled
            if self.enable_hand_detection:
                hand_results = self.hands.process(frame_rgb)
                self.last_hand_landmarks = hand_results.multi_hand_landmarks
                hand_landmarks = self.last_hand_landmarks

            if self.enable_face_detection:
                face_results = self.face_mesh.process(frame_rgb)
                self.last_face_landmarks = face_results.multi_face_landmarks if face_results.multi_face_landmarks else []
                face_landmarks = self.last_face_landmarks

            if self.enable_pose_detection:
                pose_results = self.pose.process(frame_rgb)
                self.last_pose_landmarks = pose_results.pose_landmarks
                pose_landmarks = self.last_pose_landmarks
        else:
            # Use cached results
            if self.enable_hand_detection:
                hand_landmarks = self.last_hand_landmarks

            if self.enable_face_detection:
                face_landmarks = self.last_face_landmarks

            if self.enable_pose_detection:
                pose_landmarks = self.last_pose_landmarks

        # Create annotation layer
        annotation_frame = processed_view.copy()

        # Process and draw detections
        results = self._process_detections(annotation_frame, hand_landmarks,
                                           face_landmarks, pose_landmarks)

        # Blend annotation layer with original frame
        alpha = 0.7
        processed_view = cv2.addWeighted(annotation_frame, alpha,
                                         processed_view, 1 - alpha, 0)

        # Add status indicators for which detections are enabled
        self._draw_detection_status(processed_view)
        self._draw_view_controls(processed_view)

        return processed_view, results

    def _draw_detection_status(self, frame):
        """Draw status indicators for which detections are enabled"""
        height, _, _ = frame.shape

        # Draw status text with colored indicators
        status_y = height - 10

        # Hand detection status
        hand_status = "ON" if self.enable_hand_detection else "OFF"
        hand_color = (0, 255, 0) if self.enable_hand_detection else (0, 0, 255)
        cv2.putText(frame, f"Hand: {hand_status} (1)", (10, status_y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 1)

        # Face detection status
        face_status = "ON" if self.enable_face_detection else "OFF"
        face_color = (0, 255, 0) if self.enable_face_detection else (0, 0, 255)
        cv2.putText(frame, f"Face: {face_status} (2)", (10, status_y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)

        # Pose detection status
        pose_status = "ON" if self.enable_pose_detection else "OFF"
        pose_color = (0, 255, 0) if self.enable_pose_detection else (0, 0, 255)
        cv2.putText(frame, f"Pose: {pose_status} (3)", (10, status_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pose_color, 1)

        # Skip frame info
        skip_info = f"Skip: {self.skip_frames} frames (+ / -)"
        cv2.putText(frame, skip_info, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)

    def _draw_view_controls(self, frame):
        """Draw zoom and pan status information"""
        height, width, _ = frame.shape

        # Draw zoom status
        cv2.putText(frame, f"Zoom: {self.zoom_factor:.1f}x (I/O)",
                    (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        # Draw pan status
        cv2.putText(frame,
                    f"Pan: X={self.pan_x:.2f}, Y={self.pan_y:.2f} (←↑→↓)",
                    (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

        # Draw reset hint
        cv2.putText(frame, "Reset view: R", (width - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
        if hand_landmarks and self.enable_hand_detection:
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
        if face_landmarks and self.enable_face_detection:
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
        if pose_landmarks and self.enable_pose_detection:
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
                results["pose_angles"] = \
                    self._calculate_pose_angles(pose_landmarks)

        return results

    def _recognize_gesture(self, hand_landmarks) -> str:
        """
        Recognize specific gestures based on hand landmark positions.
        """

        # Get finger extended status (except thumb which needs special handling)
        index_extended = self._is_finger_extended(hand_landmarks,
                                                  self.INDEX_TIP,
                                                  self.INDEX_MCP)
        middle_extended = self._is_finger_extended(hand_landmarks,
                                                   self.MIDDLE_TIP,
                                                   self.MIDDLE_MCP)
        ring_extended = self._is_finger_extended(hand_landmarks, self.RING_TIP,
                                                 self.RING_MCP)
        pinky_extended = self._is_finger_extended(hand_landmarks,
                                                  self.PINKY_TIP,
                                                  self.PINKY_MCP)

        # Special check for thumb (comparing with index finger)
        thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
        index_mcp = hand_landmarks.landmark[self.INDEX_MCP]
        wrist = hand_landmarks.landmark[self.WRIST]

        # Check if thumb is extended based on its position relative to index finger
        thumb_to_index = self._calculate_distance(thumb_tip, index_mcp)
        wrist_to_index = self._calculate_distance(wrist, index_mcp)
        thumb_extended = thumb_to_index > wrist_to_index * 0.5

        # Check for OK sign - thumb and index fingertip are close, other fingers extended
        thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.INDEX_TIP]
        distance_thumb_index = self._calculate_distance(thumb_tip, index_tip)
        # Threshold for considering fingers are touching
        if distance_thumb_index < 0.05 and middle_extended and ring_extended and pinky_extended:
            return "OK Sign"

        # Check for thumbs up/down
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Check thumb orientation
            wrist = hand_landmarks.landmark[self.WRIST]
            thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
            # If thumb is pointing up (y-coordinate is significantly smaller than wrist)
            if thumb_tip.y < wrist.y - 0.1:
                return "Thumb Up"
            if thumb_tip.y > wrist.y:
                return "Thumb Down"

        # Check for rock-on / horns
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "Rock On / Horns"

        # Number gestures (0-5)
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Number 0 (Fist)"

        if not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Number 1 (Index)"

        if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Number 2 (Victory)"

        if not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
            return "Number 3"

        if not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "Number 4"

        if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "Number 5 (Open Hand)"

        # Middle finger
        if not thumb_extended and not index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Middle"

        # Pinky
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "Pinky"

        # If no known gesture is detected
        return "Unknown Gesture"

    # If tip is further from wrist than MCP, finger is l
    def _is_finger_extended(self,
                            hand_landmarks,
                            tip_idx,
                            mcp_idx,
                            wrist_idx=0):
        """Check if a finger is extended using distance comparison"""
        tip_to_wrist = self._calculate_distance(
            hand_landmarks.landmark[tip_idx],
            hand_landmarks.landmark[wrist_idx])
        mcp_to_wrist = self._calculate_distance(
            hand_landmarks.landmark[mcp_idx],
            hand_landmarks.landmark[wrist_idx])

        # If tip is further from wrist than MCP, finger is likely extended
        return tip_to_wrist > mcp_to_wrist * 1.1

    @staticmethod
    def _calculate_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    @staticmethod
    def _calculate_angle(a, b, c) -> float:
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                 np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _calculate_pose_angles(self, landmarks) -> Dict[str, float]:
        """
        Calculate angles for key pose joints.
        """

        # Calculate key joint angles
        angles = {
            "left_elbow":
            self._calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]),
            "right_elbow":
            self._calculate_angle(
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST])
        }

        return angles


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
                        '--resize_ratio',
                        default=1.0,
                        type=float,
                        help='Ratio to resize live display')
    parser.add_argument('-i',
                        '--input_device',
                        default=None,
                        type=str,
                        help='Input device, file or strearming URL')
    parser.add_argument('-y',
                        '--YT_URL',
                        help='If input URL is youtube URL',
                        action='store_true')
    parser.add_argument('-s',
                        '--skip_frames',
                        default=0,
                        type=int,
                        help='Number of frames to skip between detections')
    args = parser.parse_args()

    # Get input device
    input_device = parse_video_device(args.input_device, YT_URL=args.YT_URL)

    # Initialize video capture
    cap = cv2.VideoCapture(input_device)

    # Check if stream opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Initialize detection system
    detector = DetectionSystem()
    detector.set_skip_frames(args.skip_frames)

    # Print controls
    print("\n=== CONTROLS ===")
    print("Detection Controls:")
    print("  1: Toggle hand detection")
    print("  2: Toggle face detection")
    print("  3: Toggle pose detection")
    print("  +: Increase frame skip")
    print("  -: Decrease frame skip")
    print("View Controls:")
    print("  I: Zoom in")
    print("  O: Zoom out")
    print("  Arrow keys (←↑→↓,hklj): Pan view")
    print("  R: Reset view (zoom and pan)")
    print("  Q: Quit\n")

    # Process frames
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from stream.")
                break

            # Resize frame
            height, width, _ = frame.shape
            if round(args.resize_ratio, 3) != 1.0:
                frame = resize_image(frame,
                                     width=width,
                                     height=height,
                                     resize_ratio=args.resize_ratio)

            # Process frame
            start_time = time.time()
            processed_frame, results = detector.process_frame(frame)

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the processed frame
            cv2.imshow('Comprehensive Detection System', processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Handle toggle keys
            if key == ord('1'):
                status = detector.toggle_hand_detection()
                print(f"Hand detection: {'ON' if status else 'OFF'}")
            elif key == ord('2'):
                status = detector.toggle_face_detection()
                print(f"Face detection: {'ON' if status else 'OFF'}")
            elif key == ord('3'):
                status = detector.toggle_pose_detection()
                print(f"Pose detection: {'ON' if status else 'OFF'}")
            # Handle frame skip adjustment
            elif key == ord('+'):
                detector.set_skip_frames(detector.skip_frames + 1)
                print(f"Skip frames: {detector.skip_frames}")
            elif key == ord('-'):
                detector.set_skip_frames(max(0, detector.skip_frames - 1))
                print(f"Skip frames: {detector.skip_frames}")
            # Handle zoom controls
            elif key == ord('i'):  # Zoom in
                zoom = detector.zoom_in()
                print(f"Zoom factor: {zoom:.1f}x")
            elif key == ord('o'):  # Zoom out
                zoom = detector.zoom_out()
                print(f"Zoom factor: {zoom:.1f}x")
            # Handle pan controls
            elif key == 81 or key == ord('h'):  # Left arrow or 'h'
                pan_x = detector.pan_left()
                print(f"Pan X: {pan_x:.2f}")
            elif key == 82 or key == ord('k'):  # Up arrow or 'k'
                pan_y = detector.pan_up()
                print(f"Pan Y: {pan_y:.2f}")
            elif key == 83 or key == ord('l'):  # Right arrow or 'l'
                pan_x = detector.pan_right()
                print(f"Pan X: {pan_x:.2f}")
            elif key == 84 or key == ord('j'):  # Down arrow or 'j'
                pan_y = detector.pan_down()
                print(f"Pan Y: {pan_y:.2f}")
            # Reset view
            elif key == ord('r'):
                zoom, pan_x, pan_y = detector.reset_view()
                print(
                    f"View reset: Zoom={zoom:.1f}x, Pan X={pan_x:.2f}, Pan Y={pan_y:.2f}"
                )
            # Quit
            elif key == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {str(e)}")

    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
