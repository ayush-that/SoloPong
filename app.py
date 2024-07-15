from flask import Flask, render_template, Response, request
import numpy as np
import cv2 as cv
import cvzone
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['POST'])
def reset():
    global ballPos, speedX, speedY, gameOver, score
    ballPos = [100, 100]
    speedX = 25
    speedY = 25
    gameOver = False
    score = [0, 0]
    return '', 204  # No Content response

cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgBg = cv.imread("assets/bg.png")
imgGO = cv.imread("assets/gameover.png")
imgBall = cv.imread("assets/ball.png", cv.IMREAD_UNCHANGED)
imgBat1 = cv.imread("assets/bat1.png", cv.IMREAD_UNCHANGED)
imgBat2 = cv.imread("assets/bat2.png", cv.IMREAD_UNCHANGED)

detector = HandDetector(detectionCon=0.8, maxHands=2)

ballPos = [100, 100]
speedX = 25
speedY = 25
gameOver = False
score = [0, 0]

def gen():
    global ballPos, speedX, speedY, gameOver, score
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv.flip(img, 1)
        imgRaw = img.copy()

        hands, img = detector.findHands(img, flipType=False)

        img = cv.addWeighted(img, 0.1, imgBg, 0.9, 0)

        if hands:
            for hand in hands:
                x, y, w, h = hand["bbox"]
                h1, w1, _ = imgBat1.shape
                y1 = y - h1 // 2
                y1 = np.clip(y1, 20, 415)

                if hand["type"] == "Left":
                    img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                    if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] += 30
                        score[0] += 1

                if hand["type"] == "Right":
                    img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                    if 1195 - 50 < ballPos[0] < 1195 and y1 < ballPos[1] < y1 + h1:
                        speedX = -speedX
                        ballPos[0] -= 30
                        score[1] += 1

        if ballPos[0] < 40 or ballPos[0] > 1200:
            gameOver = True
        if gameOver:
            img = imgGO
            cv.putText(
                img,
                str(score[1] + score[0]).zfill(2),
                (585, 360),
                cv.FONT_HERSHEY_SIMPLEX,
                2.5,
                (200, 0, 200),
                5,
            )

        else:
            if ballPos[1] >= 500 or ballPos[1] <= 10:
                speedY = -speedY

            ballPos[0] += speedX
            ballPos[1] += speedY

            img = cvzone.overlayPNG(img, imgBall, ballPos)
            cv.putText(
                img,
                str(score[0]),
                (300, 650),
                cv.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                5,
            )
            cv.putText(
                img,
                str(score[1]),
                (900, 650),
                cv.FONT_HERSHEY_COMPLEX,
                3,
                (255, 255, 255),
                5,
            )

        img[580:700, 20:233] = cv.resize(imgRaw, (213, 120))

        ret, buffer = cv.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
