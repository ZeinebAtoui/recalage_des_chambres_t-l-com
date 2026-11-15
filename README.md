# Automatic Geolocation of Telecom Chambers

An AI-powered computer vision solution for automatically detecting, segmenting, and estimating the geographical position of telecom chambers from Google Street View images.

## 1. Project Overview

Telecom chambers are essential components of telecommunications infrastructure. They provide access to underground cables and facilitate network maintenance and connections.

However, the positions stored in Geographic Information Systems (GIS) do not always accurately reflect their actual locations in the field. Correcting these discrepancies manually requires significant time and effort, especially when dealing with large-scale infrastructure networks.

This project aims to automate the geolocation correction process using computer vision, deep learning, and geometric estimation techniques.

The proposed solution analyzes Google Street View images to identify telecom chambers, extract their visible geometric features, and estimate their position and orientation in 3D space.

## 2. Project Objectives

The main objectives are to:

- Automatically detect telecom chambers in Street View images.
- Segment the visible surface of each chamber at pixel level.
- Extract geometric features, particularly the chamber's corners.
- Handle partially visible chambers using geometric constraints.
- Estimate the chamber's 3D position and orientation.
- Improve the accuracy and efficiency of the existing GIS geolocation process.

## 3. Solution Architecture

The solution follows a multi-stage computer vision pipeline.

```text
GIS Reference Coordinates
          |
          v
Google Street View Image Collection
          |
          v
      Faster R-CNN
     Object Detection
          |
          v
      Image Cropping
          |
          v
      DeepLabV3+
   Semantic Segmentation
          |
          v
         SAM2
  Segmentation Refinement
          |
          v
    Corner Extraction
          |
          v
 Geometric Matching
 (Constrained Template Matching)
          |
          v
  PnP / Multi-view PnP
          |
          v
 Position and Orientation Estimation
          |
          v
       Evaluation
```

Each component has a specific role:

| Component                     | Main responsibility                                     |
| ----------------------------- | ------------------------------------------------------- |
| Faster R-CNN                  | Detect telecom chambers and generate bounding boxes.    |
| DeepLabV3+                    | Produce pixel-level segmentation masks.                 |
| SAM2                          | Refine the predicted segmentation boundaries.           |
| OpenCV                        | Extract geometric features and perform pose estimation. |
| Constrained Template Matching | Match visible corners to a known geometric template.    |
| PnP / Multi-view PnP          | Estimate the chamber's position and orientation.        |

## 4. Data Preparation

### 4.1 Geographic Reference Data

The initial dataset is built from certified geographic reference points in Lyon.

**Source file:** `points_A_lyon_certifies.csv`

These coordinates serve as reference locations for collecting Google Street View images.

### 4.2 Google Street View Image Collection

For each reference point, the image collection process involves:

1. Identifying available Street View panoramas.
2. Searching for suitable viewpoints around the target location.
3. Computing the camera parameters:
   - **Bearing:** horizontal camera direction.
   - **Pitch:** vertical camera orientation.
   - **FOV:** camera field of view.
4. Downloading images at the requested zoom level.

These camera parameters are also useful during the subsequent geometric estimation stage.

### 4.3 Dataset Annotation

Two annotation formats are prepared for the detection and segmentation models.

**Object Detection — Faster R-CNN**

Bounding boxes are used to identify telecom chambers and false-positive objects.

Classes:

- `0`: Background
- `1`: False Positive
- `2`: Telecom Chamber

**Semantic Segmentation — DeepLabV3+**

Pixel-level masks are used to distinguish the relevant regions in each image.

Classes:

- `0`: Background
- `1`: False Positive
- `2`: Telecom Chamber

## 5. Deep Learning Models

### 5.1 Faster R-CNN — Object Detection

Faster R-CNN is used as the first stage of the pipeline.

It identifies candidate regions containing telecom chambers and generates bounding boxes that are subsequently used to extract image crops.

The model was selected after initial experiments showed insufficient precision with YOLO for this particular dataset.

### 5.2 DeepLabV3+ — Semantic Segmentation

DeepLabV3+ is responsible for producing detailed segmentation masks.

The architecture uses a **ResNet-50 V1c backbone pretrained on ImageNet**.

The model performs three-class semantic segmentation:

- Background
- False Positive
- Telecom Chamber

The resulting masks provide the geometric information required for contour analysis and corner extraction.

### 5.3 SAM2 — Segmentation Refinement

SAM2 is applied after DeepLabV3+ to refine the initial segmentation masks.

Its purpose is to improve boundary delineation and obtain cleaner contours, making the subsequent geometric processing more reliable.

## 6. Geometric Processing and Position Estimation

Accurate position estimation requires more than detecting the chamber. Its geometric characteristics must also be recovered from the image.

### 6.1 Corner Extraction

The predicted segmentation masks are processed to extract the visible contour and characteristic corners of the telecom chamber.

These image coordinates provide the 2D observations needed for geometric estimation.

### 6.2 Constrained Template Matching

For partially visible chambers, some corners may be hidden by obstacles or other objects.

Constrained Template Matching uses a theoretical geometric model of the chamber to match the visible contour points and reconstruct missing corners.

The reference template is based on the chamber dimensions

The matching process helps determine whether the available geometric information is sufficient for pose estimation.

### 6.3 PnP — Perspective-n-Point

The Perspective-n-Point (PnP) algorithm estimates an object's 3D position and orientation from:

- Known 2D image coordinates of the chamber's corners.
- Corresponding 3D coordinates from the theoretical chamber model.
- Camera parameters, including bearing, pitch, and field of view.

OpenCV pose estimation methods, including `solvePnP` and IPPE, are used for this stage.

### 6.4 Multi-view PnP

Multi-view PnP extends the estimation process by combining observations from multiple images or viewpoints of the same chamber.

This approach can improve the robustness and accuracy of the estimated position when multiple valid observations are available.

## 7. Implementation Scope

The developed processing pipeline focuses on the following visibility situations:

- **Case 1 — Fully visible chamber:** The chamber is detected, segmented, and its visible corners are extracted for geometric estimation.
- **Case 2A — Partially visible chamber with sufficient geometric information:** Constrained Template Matching is used to complete the geometric information before applying PnP.

Other situations, including more complex partial occlusions and total occlusion, are part of the broader proposed architecture but are not considered fully implemented in the current project scope.

##

## 8. Technologies and Tools

| Category                | Technologies               |
| ----------------------- | -------------------------- |
| Programming Language    | Python                     |
| Deep Learning           | PyTorch                    |
| Object Detection        | Faster R-CNN               |
| Semantic Segmentation   | MMSegmentation, DeepLabV3+ |
| Segmentation Refinement | SAM2                       |
| Computer Vision         | OpenCV                     |
| Geometric Estimation    | solvePnP, IPPE             |
| Image Source            | Google Street View API     |
| Backend                 | FastAPI                    |
| Frontend                | Angular                    |
| Development Environment | Linux / WSL                |

##

## 9. Conclusion

This project explores how deep learning and geometric computer vision can be combined to automate the geolocation of telecom infrastructure.

By integrating object detection, semantic segmentation, segmentation refinement, geometric matching, and PnP-based pose estimation, the solution provides a structured approach to extracting spatial information from Street View images.

Beyond the individual models, the main contribution lies in combining visual understanding and geometric reasoning to address a real-world geolocation problem.

---

**Project:** Automatic Geolocation of Telecom Chambers
**Organization:** Sofrecom
**Domain:** Artificial Intelligence, Computer Vision, Deep Learning and Geometric Estimation
