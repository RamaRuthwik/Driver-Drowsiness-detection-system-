from scipy.spatial import distance as dist
from imutils.video import FileVideoStream # type: ignore
from imutils.video import VideoStream # type: ignore
from imutils import face_utils # type: ignore
import numpy as np
import imutils # type: ignore
import time
import dlib
import cv2
import pygame
from pygame import mixer


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    # Compute eye aspect ratio
    ear = (A + B) / (2 * C)

    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])

    mar = (A + B + C) / 3

    return mar

# Set the path to the shape predictor file
shape_predictor_path = "C:/Users/Rama Ruthwik/Desktop/Drowsiness-Detection-Using-Facial-Images-master/shape_predictor_68_face_landmarks.dat"

# Set the path to the input video file (if needed)
video_path = ""

EYE_AR_THRESH = 0.23  # threshold for blink
EYE_AR_CONSEC_FRAMES = 25  # consecutive considered true
sleep_flag = 0
yawn_flag = 0
count_mouth = 0
counter = 0
total = 0
total_yawn = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()  # You can adjust the src parameter if needed
time.sleep(1.0)

sound_path = "alarm.wav"

def soundAlert(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)  # dlib’s built-in face detector.

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouth = shape[mStart: mEnd]
        mouthEAR = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        if mouthEAR > 30:
            count_mouth += 1
            if count_mouth >= 10:
                if yawn_flag < 0:
                    print("You are yawning")
                    yawn_flag = 1
                    total_yawn += 1
                    cv2.putText(frame, "Yawn Detected", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    yawn_flag = 1
            else:
                yawn_flag = -1
        else:
            count_mouth = 0
            yawn_flag = -1

        if ear < EYE_AR_THRESH:
            counter += 1
            if counter >= EYE_AR_CONSEC_FRAMES:
                if sleep_flag < 0:
                    print("You are sleeping.")
                    cv2.putText(frame, "Sleep Detected", (150, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    sleep_flag = 1
                    total += 1
            else:
                sleep_flag = -1
        else:
            counter = 0
            sleep_flag = -1

        cv2.putText(frame, "Total Sleeps: {}".format(total), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Total Yawns: {}".format(total_yawn), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mouthEAR), (540, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if total + total_yawn > 4:
            soundAlert(sound_path)
            total = 0
            total_yawn = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
  

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



