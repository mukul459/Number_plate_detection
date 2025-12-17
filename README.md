# Automatic Number Plate Recognition (ANPR)

This project implements an **Automatic Number Plate Recognition (ANPR)** system using a **Deep Learning object detection model** and **EasyOCR** for text recognition. The pipeline detects the vehicle plate region and extracts the alphanumeric text with high confidence.

---

## 📂 Input Information

- **Kaggle image data**

![ANPR Output]([https://example.com/output.jpg](https://drive.google.com/file/d/1npNDb-SXaqdIyAL0Xo_0jY9C6uyG3PII/view?usp=sharing))
---

## 🤖 Deep Learning Object Detection

- **Detected Object Class:** `Number plate`
- **Total Objects Detected:** `1`

### ⏱️ Model Performance

| Processing Stage | Time |
|------------------|------|
| Preprocessing   | 30.4 ms |
| Inference       | 7114.0 ms |
| Postprocessing  | 73.1 ms |

- **Input Shape:** `(1, 3, 384, 640)`
- **Execution Device:** CPU

---

## 🔍 Optical Character Recognition (OCR)

- **OCR Engine:** EasyOCR
- **Compute Mode:** CPU  

> ⚠️ CUDA / MPS not available — running in CPU mode  
> For faster execution, GPU acceleration is recommended

- **Automatically Downloaded Models**
- Text detection model
- Text recognition model

---

## 📌 Detection Output

### ✅ License Plate Detection

```text
Detected Plate Number: MP33C3370
Confidence Score: 0.99
Bounding Box: (0, 3, 240, 119)
