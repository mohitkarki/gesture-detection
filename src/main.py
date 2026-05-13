# HAND SIGN DETECTION USING OPENCV + CVZONE + TENSORFLOW

import cv2                         # OpenCV -> webcam & image processing
import numpy as np                 # Array & image handling
import math                        # Mathematical calculations
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


# CAMERA INITIALIZATION

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load trained AI model + labels
classifier = Classifier(
    "Model/keras_model.h5",
    "Model/labels.txt"
)

offset = 20              # Extra space around hand crop
imgSize = 300            # Model input size (300x300)

# Labels used during training
labels = ["A", "B", "C"]

# MAIN LOOP

while True:
    
    success, img = cap.read()  # READ FRAME FROM WEBCAM

    # Safety check - If webcam fails to capture frame
    if not success or img is None:
        print("Failed to capture image from webcam")
        continue

    # Create copy of original image
    # We draw bounding boxes on this image
    imgOutput = img.copy()

    # DETECT HANDS
    hands, img = detector.findHands(img)

    # If at least one hand detected
    if hands:
 
        hand = hands[0]            # Take first detected hand
        x, y, w, h = hand['bbox']  # Get bounding box coordinates

        # CREATE WHITE BACKGROUND IMAGE
        # Neural network expects fixed-size clean image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # SAFE CROPPING - Prevent crash if hand goes outside frame
        h_img, w_img, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)

        x2 = min(w_img, x + w + offset)
        y2 = min(h_img, y + h + offset)
 
        imgCrop = img[y1:y2, x1:x2]  # Crop hand image

        # Prevent empty image crash
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        # CASE 1 : HAND IS TALL
        if aspectRatio > 1:

            k = imgSize / h                                   # Scale height to 300
            wCal = math.ceil(k * w)                           # Calculate new width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize cropped image
            wGap = math.ceil((imgSize - wCal) / 2)            # Center image horizontally
            imgWhite[:, wGap:wCal + wGap] = imgResize         # Place resized image on white background

        # CASE 2 : HAND IS WIDE
        else:

            k = imgSize / w  # Scale width to 300
            hCal = math.ceil(k * h)  # Calculate new height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize image          
            hGap = math.ceil((imgSize - hCal) / 2)  # Center image vertically
            imgWhite[hGap:hCal + hGap, :] = imgResize  # Place image on white background

        # PREDICTION
        prediction, index = classifier.getPrediction(
            imgWhite,
            draw=False
        )

        print(prediction, index)  # Print prediction scores

        # DRAW RESULT ON SCREEN
        # Label background rectangle
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 60),
            (x - offset + 120, y - offset),
            (255, 0, 255),
            cv2.FILLED
        )

        # Predicted label text
        # Label background rectangle
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 60),
            (x - offset + 120, y - offset),
            (255, 0, 255),
            cv2.FILLED
        )

        # Get text size
        text = labels[index]

        (textWidth, textHeight), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            2
        )

        # Rectangle dimensions
        boxWidth = 120
        boxHeight = 60

        # Calculate centered text position
        textX = (x - offset) + (boxWidth - textWidth) // 2
        textY = (y - offset - 60) + (boxHeight + textHeight) // 2

        # Draw text
        cv2.putText(
            imgOutput,
            text,
            (textX, textY),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2
        )

        # Bounding box around hand
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (255, 0, 255),
        )

        # DEBUG WINDOWS
        cv2.imshow("Cropped Hand", imgCrop)
        cv2.imshow("Processed Image", imgWhite)

    # SHOW FINAL OUTPUT WINDOW
    cv2.imshow("Hand Sign Detection", imgOutput)

    # Press Q to quit
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# CLEANUP

cap.release()
cv2.destroyAllWindows()