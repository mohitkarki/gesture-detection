# cmd: uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image, ImageOps   # Used to fix mobile orientation
import io

app = FastAPI()

# STEP 0: LOAD MODEL + HAND DETECTOR
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

imgSize = 300
offset = 20
labels = ["A", "B", "C"]

# STEP 1: HOME ROUTE (TEST API)
@app.get("/")
def home():
    return {"message": "API is working"}

# STEP 2: PREDICTION API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # STEP 3: READ IMAGE FROM FLUTTER
        contents = await file.read()

        try:
            # Fix orientation from mobile (EXIF issue)
            image = Image.open(io.BytesIO(contents))
            image = ImageOps.exif_transpose(image)

            # Convert PIL image → OpenCV format (BGR)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        except Exception as e:
            print("Image read error:", e)
            return {"prediction": "Invalid image"}

        # STEP 4: FIX FRONT CAMERA MIRROR (VERY IMPORTANT)
        # Front camera flips image → undo it
        img = cv2.flip(img, 1)

        # STEP 5: FIX ROTATION (BASED ON YOUR CASE)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Debug: Save received image
        cv2.imwrite("flutter_input.jpg", img)

        # STEP 6: DETECT HAND
        hands, img = detector.findHands(img)

        if not hands:
            return {"prediction": "No hand detected"}

        hand = hands[0]
        x, y, w, h = hand['bbox']

        if w == 0 or h == 0:
            return {"prediction": "Invalid bounding box"}

        # STEP 7: SAFE CROPPING (NO CRASH AT EDGES)
        h_img, w_img, _ = img.shape

        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(w_img, x + w + offset)
        y2 = min(h_img, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            return {"prediction": "Invalid crop"}

        # STEP 8: CREATE WHITE BACKGROUND (MODEL INPUT)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        aspectRatio = h / w

        # STEP 9: RESIZE + CENTER IMAGE (IMPORTANT)
        if aspectRatio > 1:
            # Hand is tall (height > width)
            k = imgSize / h
            wCal = math.ceil(k * w)

            if wCal <= 0:
                return {"prediction": "Resize error"}

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            # Hand is wide (width > height)
            k = imgSize / w
            hCal = math.ceil(k * h)

            if hCal <= 0:
                return {"prediction": "Resize error"}

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Debug: Final image sent to model
        cv2.imwrite("final_input.jpg", imgWhite)

        # STEP 10: PREDICTION
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        return {
            "prediction": labels[index],
            "confidence": float(max(prediction))
        }

    except Exception as e:
        print("Server Error:", e)
        return {"prediction": "Server error"}