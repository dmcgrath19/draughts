# Computer Vision - Draughts Video and Image Processing  
### Recovered from a 2022 project
## Overview
This project implements a system to automatically analyze a game of draughts (checkers), across five parts. It includes pixel classification, piece detection, video processing, corner detection, and King piece classification.

## Table of Contents
- [Part 1 - Pixel Classification](#part-1---pixel-classification)
- [Part 2 - Piece Detection](#part-2---piece-detection)
- [Part 3 - Video Processing](#part-3---video-processing)
- [Part 4 - Edge and Corner Detection](#part-4---edge-and-corner-detection)
- [Part 5 - King Piece Detection](#part-5---king-piece-detection)

---

## Part 1 - Pixel Classification

### Problem Statement
Classify each pixel in an image into one of:
- White Piece
- Black Piece
- White Square
- Black Square
- None of the above

### Approach
- Convert BGR image to HSV
- Perform back projection with sample histograms
- Convert result to grayscale and threshold it
- Use morphological operations to clean the binary mask
- Apply Connected Components Analysis (CCA) for region detection

### Results

| Original Image | HSV Classification | Ground Truth |
|----------------|---------------------|--------------|
| ![](/pictures/SampleImages/DraughtsGame1Move0.png) | ![](/pictures/SampleImages/HSV-Combined-Image.png) | ![](/pictures//SampleImages/DraughtsGame1Move0GroundTruth.png) |

### Remarks
- Board detection is generally accurate
- Misclassifications outside the board (e.g., table and frame)
- Process depends heavily on consistent lighting

---

## Part 2 - Piece Detection

### Problem Statement
Detect pieces on each square using known corner locations and classify them by color.

### Method
- Apply a perspective transform to isolate the board
- Use CCA results from Part 1 to detect pieces
- Clean classification by verifying if detected pieces fall on expected square types
- Generate PDN matrix from center point checks

### Results

| Bad Detection & Improved Detection |
|-----------------------------------|
| ![](/pictures/SampleImages/Bad-v-Good-piece-detection.png) |

- **Accuracy:** 97% on static images  
- 753 True Positives, 66 False Positives, 0 False Negatives

### Remarks
- Cleaning classification greatly improves accuracy
- Center pixel check for PDN simplifies logic and improves efficiency

---

## Part 3 - Video Processing

### Problem Statement
Detect valid move frames and classify moves over time from a draughts video.

### Method
- Perspective transform for every frame
- Apply Gaussian Mixture Model (GMM) for motion detection
- Detect frames with <3% motion for analysis
- Use piece detection logic from Part 2 to track moves

### Results
- 76 frames analyzed (vs 68 Ground Truth)
- 84% move detection accuracy (57/68)
- Some failures due to shadows obscuring pieces

| GMM Example |
|-------------|
| ![](/pictures/GMMEg.png) |

### Remarks
- GMM is effective, but sensitive to background modeling errors
- Shadows and lighting still affect detection

---

## Part 4 - Edge and Corner Detection

### Methods Compared
1. **Hough Transform:** Effective for long lines but noisy
2. **Contour Segmentation:** Small segments, good for extracting board pattern
3. **OpenCV `findChessboardCorners()`:** Reliable but sensitive to dark borders


| Hough Lines |
|---------------|
| ![](/pictures/HoughLines.png) |

| Line Segments |
|---------------|
| ![](/pictures/LineSegments.png)|

| findChessboardCorners |
|------------------------|
| ![](/pictures/FindChessboardCorners.png) |

### Remarks
- Combining Hough + contour filtering could improve corner detection
- OpenCV's method limited by board design

---

## Part 5 - King Piece Detection

### Problem Statement
Detect and differentiate King pieces from regular pieces.

### Attempted Solutions
- Simple heuristic: piece in final row becomes King
- Proposed: Use shape features (elongatedness, rectangularity)
- Proposed: Filter shapes via Hough Circles

| Hough Circles |
|---------------|
| ![](/pictures/HoughCircles.png) |

### Results
- Improved detection in specific cases (static kings)
- Realistic performance still low due to limited positional logic
- Errors arise from perspective distortion and background interference

### Final Remarks
- Strong performance on static image detection and board classification
- Video tracking and King detection need robustness improvements
- Future improvements: dynamic King tracking, lighting invariant features, adaptive background modeling

---

## Requirements
- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib (for visualizations)

---

## How to Run
1. Clone the repo
2. Run each part's script:
   ```bash
   python part1_pixel_classification.py
   python part2_piece_detection.py
   python part3_video_processing.py
   # etc.
```
3. (optional) take images/videos and place in folder
