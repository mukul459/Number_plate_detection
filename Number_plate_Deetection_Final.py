pip install easyocr
pip install pytesseract
pip install ultralytics
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import os
import re
import pytesseract

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    gray = cv2.fastNlMeansDenoising(gray, h=10)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)



    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)


    sharpened = cv2.resize(sharpened, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh
def ocr_plate(plate_img):

    config = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng --oem 3 --psm 7"
    text = pytesseract.image_to_string(plate_img, config=config)

    text = text.strip()
    text = text.replace(" ", "").replace("\n", "")

    text = "".join([c for c in text if c.isalnum()])
    return text

from ultralytics import YOLO
model = YOLO('yolo11x.pt')

def find_number_plate_dl(img_path, image):
  results = model(img_path)
  plate_crops_and_bboxes = []

  for r in results:
      for box in r.boxes:
          x1, y1, x2, y2 = map(int, box.xyxy[0])
          cropped_img = image[y1:y2, x1:x2]
          plate_crops_and_bboxes.append((cropped_img, (x1, y1, x2-x1, y2-y1)))
  return plate_crops_and_bboxes

def extract_license_number(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    plate_crops_and_bboxes = find_number_plate_dl(image_path, img)
    final_results = []

    reader = easyocr.Reader(['en'])

    for item in plate_crops_and_bboxes:
        if not (isinstance(item, tuple) and len(item) == 2):
            print(f"Warning: Skipping malformed item in plate_crops_and_bboxes. Expected (image, bbox), got {item}")
            continue
        cropped_img, bbox = item


        easyocr_results = reader.readtext(cropped_img)
        detected_text_easyocr = []
        confidences_easyocr = []
        for (bbox_ocr, text_ocr, prob_ocr) in easyocr_results:
            detected_text_easyocr.append(text_ocr)
            confidences_easyocr.append(prob_ocr)

        full_text_easyocr = "".join(detected_text_easyocr).replace(" ", "").replace("\n", "").upper()
        avg_confidence_easyocr = np.mean(confidences_easyocr) if confidences_easyocr else 0.0


        pytesseract_text = ocr_plate(cropped_img)
        pytesseract_text = "".join([c for c in pytesseract_text if c.isalnum()]).upper()

        selected_text = ""
        selected_confidence = 0.0


        if avg_confidence_easyocr >= 0.6:
            selected_text = full_text_easyocr
            selected_confidence = avg_confidence_easyocr
        elif pytesseract_text and len(pytesseract_text) > 3 and pytesseract_text.isalnum():
            selected_text = pytesseract_text


            selected_confidence = avg_confidence_easyocr if avg_confidence_easyocr > 0 else 0.4
        else:
            selected_text = full_text_easyocr
            selected_confidence = avg_confidence_easyocr


        if selected_confidence >= 0.4:
            final_results.append((selected_text, bbox, selected_confidence))



        plt.imshow(cropped_img)
        plt.title(f"Detected: {selected_text} (Conf: {selected_confidence:.2f})")
        plt.show()

    final_results.sort(key=lambda x: x[2], reverse=True)

    if final_results:
        print("\n--- DL Model Detections ---")
        for text, bbox, conf in final_results:
            print(f"Detected: {text} (Confidence: {conf:.2f}) at BBox: {bbox}")

    return final_results

if __name__ == "__main__":
    folder_path = r"/content/sample_data"


    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Processing images in folder: {folder_path}")
    print(f"Found image files: {image_files}")


    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcessing: {image_path}")

        matches = extract_license_number(image_path)

        if matches:
            for txt, bbox, confidence in matches:
                print(f"Final Detected Plate: {txt} (Confidence: {confidence:.2f}) BBox: {bbox}")
        else:
            print(f"No plate detected with sufficient confidence in {filename}")