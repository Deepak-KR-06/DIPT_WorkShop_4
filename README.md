# Workshop 4: Gray Scale Morphology - Coin Detection

**Name:** Deepak K R

**Register No:** 212225040057

---

## Aim

To implement grayscale morphological operations (erosion and dilation) combined with Canny edge detection to identify and highlight circular coin boundaries in an input image.

---

## Software Required

* **Python 3.x**
* **OpenCV (`opencv-python`)**
* **NumPy (`numpy`)**
* **Matplotlib (`matplotlib`)**

---

## Algorithm

1. **Image Loading:**
* Read the input image `'CoinsA.png'` using `cv2.imread()`.
* Check if the image loaded successfully; if not, print an error message and exit.


2. **Preprocessing:**
* Convert the input image from BGR to grayscale using `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
* Apply Gaussian blurring via `cv2.GaussianBlur(gray, (5, 5), 0)` to smooth noise while preserving coin edges.


3. **Morphological Operations:**
* Create a $5 \times 5$ rectangular structuring element (kernel) filled with ones.
* Apply **Erosion** (`cv2.erode`) to reduce minor noise and isolate coin features.
* Apply **Dilation** (`cv2.dilate`) on the eroded image to restore coin region sizes and smooth outer boundaries.


4. **Edge & Contour Detection:**
* Detect edges using the **Canny Edge Detector** (`cv2.Canny`) with thresholds set to 50 and 150.
* Extract outer contours of the coins using `cv2.findContours` with the `RETR_EXTERNAL` flag and `CHAIN_APPROX_SIMPLE` approximation.


5. **Visualization:**
* Overlay detected contours onto a copy of the original image using `cv2.drawContours` with a green border (color `(0, 255, 0)` and thickness 2).
* Convert BGR color channels to RGB for both original and processed images.
* Plot and display both images side-by-side using `matplotlib.pyplot`.



---
## Program

```py
# WORKSHOP 4 - Gray Scale Morphology - Real-Time Bone Fracture Detection
# Name : Deepak K R
# Reg No : 212225040057

import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def detect_fractures(preprocessed, original):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(preprocessed, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    edges = cv2.Canny(dilation, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = original.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

def present_results(original_image, processed_image):
    # Convert from BGR (OpenCV) to RGB (Matplotlib)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Fracture Detected Image")
    plt.imshow(processed_rgb)
    plt.axis('off')

    plt.show()

# --- Main Execution ---
image_path = 'CoinsA.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Check the file path.")
else:
    preprocessed = preprocess_image(image)
    fracture_detected_image = detect_fractures(preprocessed, image)
    present_results(image, fracture_detected_image)
```
---
## Output

### Input vs Detected Image

<img width="1102" height="575" alt="image" src="https://github.com/user-attachments/assets/a2a5aac8-b74d-42cd-a3a9-fed21e8dec2c" />

---

## Result

The program for coin detection using grayscale morphology and edge detection techniques has been successfully implemented and executed. All requested README contents (Aim, Software Required, Algorithm, Output placeholder, and Result) are complete.
