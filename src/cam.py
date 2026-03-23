import cv2

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()