# Image Classification for HSP Method

**Version:** v01.63  
**Author:** Manazael Zuliani Jora  
**Date:** Dec/15/2025

## 📋 Overview

This project provides a specialized algorithm (`image_classification_for_hsp_method`) designed to analyze and classify images of chemical formulations based on **turbidity** and **phase separation** phenomena.

Unlike "black-box" Machine Learning approaches, this algorithm uses **statistical pixel analysis** (intensity profiling, histograms, derivatives, and contrast logic). It was developed specifically to handle small datasets where training a Convolutional Neural Network (CNN) is not yet viable, providing robust analytical metrics to aid human classification.

## 🎯 Business Rules & Classification Logic

The algorithm aids in classifying formulations into 6 distinct categories based on pixel intensity distribution ($0-255$) and vertical homogeneity:

| Class | Description | Physical Phenomenon | Visual Characteristics |
| :--- | :--- | :--- | :--- |
| **1 - 4** | **Turbidity Levels** | From clear (1) to turbid (4). | Analyzed via global histogram distribution and contrast reduction. As turbidity increases, the histogram "shrinks" towards gray, reducing dynamic range. |
| **5** | **Heterogeneous Dispersion** | Phase separation starting to form a dispersion. | Characterized by high variance in pixel intensity without a clear vertical pattern. |
| **6** | **Phase Separation** | Sedimentation, Creaming, or Flotation. | Identified by sharp peaks in the **Vertical Profile Derivative**, indicating a sudden change in intensity at a specific height. |

### Statistical Concepts Used
* **P10 ($P_{min}$)**: Represents the "floor" of the formulation's darkness, excluding the darkest 10% of pixels (sensor noise or extreme shadows).
* **P90 ($P_{max}$)**: Represents the "ceiling" of the formulation's brightness, excluding the brightest 10% of pixels (specular highlights/glass reflections).
* **P50 (Median)**: The robust central tendency of the formulation's appearance.
* **Absolute Contrast Normalization**: Contrast is calculated by dividing by **255** (sensor capacity) rather than the dynamic range ($P_{100}-P_0$). This ensures that "washed out" (turbid) images correctly report low contrast, rather than artificially high relative contrast.

## 🛠 Features & Visual Output

The function generates two distinct analytical objects per image:

### Object 1: Visual & Vertical Profile Analysis
A composed figure (1x4 grid) specifically designed with zero whitespace ("glued" layout) to correlate visual data directly with analytical data pixel-by-pixel.
1.  **Original RGB**: The raw input image.
2.  **Grayscale**: The converted luminance channel used for calculations.
3.  **Contrast Overlay**: A diagnostic mask:
    * **Dark Red**: Pixels $< P_{min}$ (Shadows).
    * **Light Red**: Pixels $> P_{max}$ (Highlights).
4.  **Vertical Profile Graph**:
    * **Blue Curves**: $P_{min}$, $P_{50}$, and $P_{max}$ intensity per row height.
    * **Red Curve**: Derivative of $P_{50}$ (crucial for identifying Class 6 phase boundaries).
    * **X-Axis**: Pixel Intensity (Blue).
    * **Top Axis**: Derivative Magnitude (Red).

### Object 2: Histogram Analysis
A statistical distribution of the entire formulation.
* **Bars**: Pixel count per intensity ($0-255$).
* **Red Curve**: Cumulative Percentage (0-100%).
* **Vertical Markers**: User-defined thresholds for Classes 1, 2, 3, and 4.
* **Metrics**: Displays calculated contrast percentages in the legend.

## 📦 Dependencies

* `numpy`
* `matplotlib`
* `Pillow` (PIL)
* `scipy`

## 🚀 Usage

### Function Arguments
```python
image_classification_for_hsp_method_v01_63(
    image_paths,                                # List of image file paths
    show_plots=True,                            # Boolean to render charts
    minimum_percentil=10,                       # Lower bound for statistical exclusion
    maximum_percentil=90,                       # Upper bound for statistical exclusion
    maximum_pixel_intensity_for_class_1=75,     # Threshold for Class 1
    maximum_pixel_intensity_for_class_2=110,    # Threshold for Class 2
    maximum_pixel_intensity_for_class_3=150,    # Threshold for Class 3
    maximum_pixel_intensity_for_class_4=255     # Threshold for Class 4
)
