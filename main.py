import sys
import os
import time
import cv2
import mediapipe as mp
import pyautogui
import csv
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon

COOLDOWN_SECONDS = 1.5
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "gesture_log.csv")


class GestureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control Pro")
        if os.path.exists("icons/logo.png"):
            self.setWindowIcon(QIcon("icons/logo.png"))
        self.setGeometry(300, 150, 820, 640)

        main_layout = QVBoxLayout()
        title = QLabel("🤖 Gesture Control Pro", alignment=Qt.AlignCenter)
        title.setStyleSheet("font-size:20px; font-weight:bold; color:#00d9ff;")
        main_layout.addWidget(title)

        middle = QHBoxLayout()

        self.camera_label = QLabel(alignment=Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black;")
        middle.addWidget(self.camera_label)

        right_col = QVBoxLayout()
        self.gesture_display = QLabel("No gesture detected", alignment=Qt.AlignTop)
        self.gesture_display.setFixedWidth(140)
        self.gesture_display.setWordWrap(True)
        self.gesture_display.setStyleSheet("font-size:14px;")
        right_col.addWidget(self.gesture_display)

        self.cooldown_label = QLabel("Ready", alignment=Qt.AlignTop)
        right_col.addWidget(self.cooldown_label)
        right_col.addStretch()
        middle.addLayout(right_col)
        main_layout.addLayout(middle)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Camera")
        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.6)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.running = False

        self.last_action_time = 0
        self.prev_gestures = {"Left": None, "Right": None, "Face": None}
        self.gesture_start_time = {}

        os.makedirs(LOG_DIR, exist_ok=True)
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Hand", "Gesture", "Action", "Confidence", "Duration(s)"])

    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.2)
        if not self.cap or not self.cap.isOpened():
            self.gesture_display.setText("Camera not found")
            return
        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.last_action_time = time.time() - COOLDOWN_SECONDS
        self.timer.start(30)

    def stop_camera(self):
        self.running = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_label.clear()
        self.gesture_display.setText("Camera stopped")
        self.cooldown_label.setText("Ready")

    def update_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = self.face_mesh.process(rgb_frame)
        face_gesture = None
        face_action = None
        face_confidence = 0.0
        if face_results.multi_face_landmarks:
            fg, fa = self.detect_face_gesture(face_results.multi_face_landmarks[0], frame.shape)
            if fg:
                face_gesture, face_action = fg, fa
                face_confidence = 1.0

        hand_results = self.hands.process(rgb_frame)
        display_texts = []
        actions_performed = [] 

        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_label = hand_results.multi_handedness[idx].classification[0].label  
                confidence = hand_results.multi_handedness[idx].classification[0].score

                gesture, action = self.detect_gesture(hand_landmarks)
                wrist = hand_landmarks.landmark[0]
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])
                cv2.rectangle(frame, (max(10, x-120), max(10, y-60)), (min(frame.shape[1]-10, x+140), y+10), (0,0,0), -1)
                cv2.putText(frame, f"{hand_label} Hand: {gesture}", (max(12, x-116), max(22, y-40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Action: {action}", (max(12, x-116), max(42, y-20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                current_time = time.time()
                if gesture != self.prev_gestures.get(hand_label) and (current_time - self.last_action_time) > COOLDOWN_SECONDS:
                    if action and action != "None":
                        if action not in actions_performed:
                            self.perform_action(action)
                            duration = 0.0
                            if hand_label in self.gesture_start_time:
                                duration = round(current_time - self.gesture_start_time.get(hand_label, current_time), 2)
                            self.log_gesture(hand_label, gesture, action, confidence, duration)
                            actions_performed.append(action)
                            self.last_action_time = current_time
                            self.gesture_start_time[hand_label] = current_time
                    self.prev_gestures[hand_label] = gesture

                display_texts.append(f"{hand_label}: {gesture}")

        if face_gesture:
            cv2.rectangle(frame, (10, 10), (360, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Face: {face_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Action: {face_action}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            current_time = time.time()
            if face_gesture != self.prev_gestures.get("Face") and (current_time - self.last_action_time) > COOLDOWN_SECONDS:
                if face_action and face_action != "None":
                    self.perform_action(face_action)
                    self.log_gesture("Face", face_gesture, face_action, face_confidence, 0.0)
                    self.prev_gestures["Face"] = face_gesture
                    self.last_action_time = current_time

            display_texts.append(f"Face: {face_gesture}")

        hud_text = " | ".join(display_texts) if display_texts else "No gesture detected"
        overlay = frame.copy()
        alpha = 0.4
        cv2.rectangle(overlay, (0, frame.shape[0]-70), (frame.shape[1], frame.shape[0]), (10, 10, 10), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, hud_text, (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cd_elapsed = time.time() - self.last_action_time
        if cd_elapsed < COOLDOWN_SECONDS:
            self.cooldown_label.setText(f"Cooldown: {round(COOLDOWN_SECONDS - cd_elapsed,1)}s")
        else:
            self.cooldown_label.setText("Ready")

        self.gesture_display.setText(hud_text)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(pixmap)

    def detect_gesture(self, hand_landmarks):
        lm = hand_landmarks.landmark
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        fingers.append(1 if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x else 0)
        for id in range(1, 5):
            fingers.append(1 if lm[tip_ids[id]].y < lm[tip_ids[id] - 2].y else 0)
        total = fingers.count(1)

        if total == 0:
            return "Fist ✊", "Minimize All"
        if total == 5:
            return "Open Palm 🖐", "Play/Pause Media"
        if total == 1 and fingers[1] == 1:
            return "Point ☝️", "Volume Up"
        if total == 2 and fingers[1] == fingers[2] == 1:
            return "Peace ✌️", "Volume Down"
        if fingers == [1, 0, 0, 0, 0]:
            return "Thumbs Up 👍", "Increase Brightness"
        if fingers == [0, 0, 0, 0, 1]:
            return "Thumbs Down 👎", "Decrease Brightness"
        return f"{total} Fingers", "None"

    def detect_face_gesture(self, face_landmarks, frame_shape):
        lm = face_landmarks.landmark
        h = frame_shape[0]
        left_eye_top = lm[159]
        left_eye_bottom = lm[145]
        right_eye_top = lm[386]
        right_eye_bottom = lm[374]
        mouth_top = lm[13]
        mouth_bottom = lm[14]
        left_eye_dist = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_dist = abs(right_eye_top.y - right_eye_bottom.y)
        mouth_open_dist = abs(mouth_top.y - mouth_bottom.y)
        blink_threshold = 0.012
        mouth_threshold = 0.04
        if left_eye_dist < blink_threshold and right_eye_dist < blink_threshold:
            return "Blink 👁️", "Play/Pause Media"
        if mouth_open_dist > mouth_threshold:
            return "Mouth Open 😮", "Mute Volume"
        return None, None

    def perform_action(self, action):
        try:
            if action == "Minimize All":
                pyautogui.hotkey("win", "d")
            elif action == "Play/Pause Media":
                pyautogui.press("playpause")
            elif action == "Volume Up":
                pyautogui.press("volumeup")
            elif action == "Volume Down":
                pyautogui.press("volumedown")
            elif action == "Increase Brightness":
                pyautogui.hotkey("fn", "brightnessup")
            elif action == "Decrease Brightness":
                pyautogui.hotkey("fn", "brightnessdown")
            elif action == "Mute Volume":
                pyautogui.press("volumemute")
        except Exception as e:
            print("Action error:", e)

    def log_gesture(self, hand, gesture, action, confidence, duration):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, hand, gesture, action, round(confidence, 2), duration])
            print(f"[LOG] {timestamp} - {hand} - {gesture} -> {action}")
        except Exception as e:
            print("Logging error:", e)

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureApp()
    window.show()
    sys.exit(app.exec_())
