import cv2                     # OpenCV -> camera handling & image processing
import numpy as np             # Used for image arrays and creating blank images
import math                    # Mathematical operations (ceil for rounding)
from cvzone.HandTrackingModule import HandDetector   # Detects hand using MediaPipe
from cvzone.ClassificationModule import Classifier   # Loads trained AI model

# CAMERA & MODEL INITIALIZATION 
cap = cv2.VideoCapture(1)  
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") # Loads trained deep learning model + label names

offset = 20        
imgSize = 300      # Final image size required by AI model (300x300)
labels = ["A", "B", "C"]

while True:

    success, img = cap.read()             # Read frame from webcam
    imgOutput = img.copy()                # Copy original image so drawings don't affect raw frame
    hands, img = detector.findHands(img)  # Detect hands in image

    # If at least one hand is detected
    if hands:

        hand = hands[0]             # Take first detected hand
        x, y, w, h = hand['bbox']   # Bounding box around hand

        # Create white background image (300x300)
        # Neural network expects fixed-size clean input
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # SAFE CROPPING (prevents crash when hand is near edges)
        h_img, w_img, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(w_img, x + w + offset)
        y2 = min(h_img, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Prevent empty image crash
        if imgCrop.size == 0:
            continue

        imgCropShape = imgCrop.shape   # Get cropped image dimensions
        aspectRatio = h / w  # Calculate aspect ratio (height / width)

        # CASE 1: HAND IS TALL (HEIGHT > WIDTH)
        if aspectRatio > 1:

            k = imgSize / h                 # Scaling factor
            wCal = math.ceil(k * w)         # Calculate new width

            # Ensure valid resize
            if wCal <= 0:
                continue

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            # Center resized image horizontally
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

            # Send processed image to AI model for prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Print probability scores and predicted index
            print(prediction, index)

        # CASE 2: HAND IS WIDE (WIDTH > HEIGHT)
        else:

            k = imgSize / w                 # Scaling factor
            hCal = math.ceil(k * h)         # Calculate new height

            # Ensure valid resize
            if hCal <= 0:
                continue

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            # Center resized image vertically
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw filled rectangle above hand (label background)
        cv2.rectangle(
            imgOutput,
            (x-offset, y-offset-50),
            (x-offset+80, y-offset-50+50),
            (255, 0, 255),
            cv2.FILLED
        )

        # Write predicted label (A/B/C)
        cv2.putText(
            imgOutput,
            labels[index],
            (x, y-26),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2
        )

        # Draw bounding box around detected hand
        cv2.rectangle(
            imgOutput,
            (x-offset, y-offset),
            (x+w+offset, y+h+offset),
            (255, 0, 255),
            4
        )

        # Debug windows (useful while training/testing)
        cv2.imshow("ImageCrop", imgCrop)     # Cropped hand
        cv2.imshow("ImageWhite", imgWhite)   # Final AI input image

    # Show final output frame
    cv2.imshow("Image", imgOutput)

    # Show final output frame
    cv2.imshow("Image", imgOutput)

    # Press 'q' to exit program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()