import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image
import os
from datetime import datetime
from scipy.stats import norm, cumfreq

def image_classification_for_hsp_method_v01_63(
    image_paths,
    show_plots=True,
    minimum_percentil=10,
    maximum_percentil=90,
    maximum_pixel_intensity_for_class_1=75,
    maximum_pixel_intensity_for_class_2=110,
    maximum_pixel_intensity_for_class_3=150,
    maximum_pixel_intensity_for_class_4=255
):
    r"""
    Overview
    --------
    This function processes a list of image paths containing chemical formulation samples to analyze
    turbidity and phase separation. Version 01.63 addresses visual alignment issues.

    **Key Fixes:**
    1. **Height Alignment**: The figure heights of Object 1 and Object 2 have been standardized to `6.0` inches to ensure they appear with the same vertical footprint.
    2. **Image Gluing**: The explicit `anchor='E'` argument was removed from the image subplots in Object 1. Relying solely on `wspace=0.0` and `aspect='auto'` provides a more robust seamless "glued" appearance between the three images and the graph.

    Core Logic and Rules
    --------------------
    1. **Preprocessing**: Loads images in RGB and Gray. Creates a 3rd "Overlay" image matrix based on percentile thresholds.
    2. **User Parameters**:
       - `minimum_percentil` ($P_{min}$): Default 10. Range [0, 50).
       - `maximum_percentil` ($P_{max}$): Default 90. Range (50, 100].
       - Class Thresholds: Integers 0-255 defining turbidity zones.
    3. **Statistical Definitions**:
       - **P10 (or $P_{min}$)**: Excludes the darkest pixels (dark noise, shadows).
       - **P90 (or $P_{max}$)**: Excludes the brightest pixels (specular highlights).
       - **|P90 - P10|**: Represents the intensity range where the majority of the formulation data lives.
    4. **Visuals**:
       - **Object 1**: 1x4 Grid. `figsize=(20, 6)`. Images perfectly glued.
       - **Object 2**: Histogram. `figsize=(7.5, 6)`. Red Cumulative Curve.

    Parameters
    ----------
    image_paths : list of str
        List of file paths.
    show_plots : bool, optional
        Display plots.
    minimum_percentil : int, optional
        Lower bound percentile (0-49).
    maximum_percentil : int, optional
        Upper bound percentile (51-100).
    maximum_pixel_intensity_for_class_1...4 : int, optional
        Thresholds for classification lines.

    Returns
    -------
    dict
        Dictionary with metrics. Returns empty dict if validation fails.

    Validations and Checks
    ----------------------
    - Checks if `minimum_percentil` is >= 0 and < 50.
    - Checks if `maximum_percentil` is > 50 and <= 100.
    - Checks if class thresholds are within 0-255.
    - File existence check.

    Change Log
    ----------
    v01.63 - Dec/15/2025
        - Layout Fix (Heights): Standardized `figsize` height to 6.0 for both objects.
          Object 1 changed to `(20, 6)`. Object 2 changed to `(7.5, 6)`.
        - Layout Fix (Gluing): Removed `anchor='E'` from the three image subplots in Object 1. Relying on `aspect='auto'` and `wspace=0.0` improves seamless gluing between images.

    v01.62 - Dec/15/2025
        - Logic Update (Overlay Image): Swapped colors (Low=Dark Red, High=Light Red).

    v01.61 - Dec/15/2025
        - Layout Fix (Object 1): Changed `figsize` to `(24, 5)` and images to `aspect='auto'`.

    v01.60 - Dec/15/2025
        - Layout Update: Attempted pixel-based ratios.

    v01.59 - Dec/14/2025
        - Layout Update: 3 Images + Graph. Contrast Overlay.

    v01.58 - Dec/14/2025
        - Visual Update: Blue X-axis (Graph 1), Red Curve (Graph 2), Legend Sorting.

    v01.0 - Dec/14/2025
        - Initial release.

    Credits
    -------
    Developed by Manazael Zuliani Jora
    Date: Dec/15/2025

    Test Example
    ------------
    >>> imgs = ['1.png']
    >>> res = image_classification_for_hsp_method_v01_63(imgs, show_plots=True, minimum_percentil=5, maximum_percentil=95)
    """

    # --- Validations ---
    if not (0 <= minimum_percentil < 50):
        print(f"Instruction for User: 'minimum_percentil' must be an integer >= 0 and < 50. You provided {minimum_percentil}. Please adjust the parameter.")
        return {}

    if not (50 < maximum_percentil <= 100):
        print(f"Instruction for User: 'maximum_percentil' must be an integer > 50 and <= 100. You provided {maximum_percentil}. Please adjust the parameter.")
        return {}

    thresholds_check = [maximum_pixel_intensity_for_class_1, maximum_pixel_intensity_for_class_2,
                        maximum_pixel_intensity_for_class_3, maximum_pixel_intensity_for_class_4]
    for i, th in enumerate(thresholds_check, 1):
        if not (0 <= th <= 255):
            print(f"Instruction for User: 'maximum_pixel_intensity_for_class_{i}' must be between 0 and 255. You provided {th}.")
            return {}

    results = {}

    for img_path in image_paths:
        try:
            if not os.path.exists(img_path):
                print(f"Error: File not found at {img_path}")
                continue

            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                img_gray = img.convert('L')
                img_array = np.array(img_gray)

            # --- GLOBAL CALCULATIONS ---
            p_min = np.percentile(img_array, minimum_percentil)
            p_max = np.percentile(img_array, maximum_percentil)
            p50 = np.percentile(img_array, 50)
            p0 = np.min(img_array)
            p100 = np.max(img_array)

            contrast_mid_global = (abs(p_max - p_min) / 255.0) * 100
            contrast_shadow_global = (abs(p_min - p0) / 255.0) * 100
            contrast_highlight_global = (abs(p100 - p_max) / 255.0) * 100

            # --- OVERLAY IMAGE GENERATION ---
            img_overlay = np.stack((img_array,)*3, axis=-1)
            mask_low = img_array < p_min
            mask_high = img_array > p_max

            # Low Intensity (Shadows) -> Dark Red
            img_overlay[mask_low] = [139, 0, 0]
            # High Intensity (Highlights) -> Light Red
            img_overlay[mask_high] = [255, 100, 100]

            # --- ROW-WISE CALCULATIONS ---
            row_median = np.median(img_array, axis=1)
            row_p_min = np.percentile(img_array, minimum_percentil, axis=1)
            row_p_max = np.percentile(img_array, maximum_percentil, axis=1)
            row_p0 = np.min(img_array, axis=1)
            row_p100 = np.max(img_array, axis=1)

            gradient_profile = np.gradient(row_median)

            row_contrast_mid_arr = (np.abs(row_p_max - row_p_min) / 255.0) * 100
            max_contrast_mid_val = np.max(row_contrast_mid_arr)
            max_contrast_mid_h = np.argmax(row_contrast_mid_arr)

            row_contrast_shadow_arr = (np.abs(row_p_min - row_p0) / 255.0) * 100
            max_contrast_shadow_val = np.max(row_contrast_shadow_arr)
            max_contrast_shadow_h = np.argmax(row_contrast_shadow_arr)

            row_contrast_highlight_arr = (np.abs(row_p100 - row_p_max) / 255.0) * 100
            max_contrast_highlight_val = np.max(row_contrast_highlight_arr)
            max_contrast_highlight_h = np.argmax(row_contrast_highlight_arr)

            file_name = os.path.basename(img_path)
            results[file_name] = {
                'p_min_val': p_min, 'p_max_val': p_max, 'p50_val': p50,
                'contrast_mid_global': contrast_mid_global,
                'max_contrast_mid_local': max_contrast_mid_val,
                'max_contrast_mid_height': max_contrast_mid_h
            }

            if show_plots:
                # =========================================================
                # OBJECT 1: 3 IMAGES + VERTICAL PROFILE
                # =========================================================
                # UPDATED v01.63: Set height to 6 to match Object 2. Adjusted width ratio.
                fig1 = plt.figure(figsize=(20, 6), layout='constrained')
                # Pixel ratios [82, 82, 82, 255] ensure correct width proportions
                gs1 = fig1.add_gridspec(1, 4, width_ratios=[82, 82, 82, 255], wspace=0.0)
                fig1.suptitle(f'Object 1 - Vertical Profile Analysis: {file_name}', fontsize=14)

                # --- 1A: Original RGB ---
                ax_img1 = fig1.add_subplot(gs1[0])
                # UPDATED v01.63: Removed anchor='E' for better seamless gluing
                ax_img1.imshow(img_rgb, aspect='auto')
                ax_img1.set_xticks([])
                ax_img1.set_yticks([])
                ax_img1.set_title("Original RGB", fontsize=10, pad=5)
                for spine in ax_img1.spines.values(): spine.set_visible(False)

                # --- 1B: Grayscale ---
                ax_img2 = fig1.add_subplot(gs1[1])
                # UPDATED v01.63: Removed anchor='E'
                ax_img2.imshow(img_array, cmap='gray', vmin=0, vmax=255, aspect='auto')
                ax_img2.set_xticks([])
                ax_img2.set_yticks([])
                ax_img2.set_title("Grayscale Image", fontsize=10, pad=5)
                for spine in ax_img2.spines.values(): spine.set_visible(False)

                # --- 1C: Overlay (Red Zones) ---
                ax_img3 = fig1.add_subplot(gs1[2])
                # UPDATED v01.63: Removed anchor='E'
                ax_img3.imshow(img_overlay, aspect='auto')
                ax_img3.set_xticks([])
                ax_img3.set_yticks([])
                ax_img3.set_title("Contrast Overlay", fontsize=10, pad=5)
                for spine in ax_img3.spines.values(): spine.set_visible(False)

                # --- 1D: Vertical Profile Graph ---
                ax_prof = fig1.add_subplot(gs1[3])
                heights = np.arange(len(row_median))
                img_height_px = img_array.shape[0]

                # Axis Config - STRICT BLUE ENFORCEMENT
                ax_prof.set_xlabel("Pixel Intensity", color='blue')
                ax_prof.tick_params(axis='x', labelcolor='blue', colors='blue')
                ax_prof.spines['bottom'].set_color('blue')
                ax_prof.spines['bottom'].set_linewidth(1.5)

                ax_prof.set_xlim(0, 255)
                ax_prof.set_ylim(img_height_px - 0.5, -0.5)
                ax_prof.set_ylabel("Height (px)", color='black')
                ax_prof.yaxis.set_label_position("right")
                ax_prof.yaxis.tick_right()
                ax_prof.tick_params(axis='y', labelcolor='black', colors='black')

                ax_grad = ax_prof.twiny()
                ax_grad.set_xlabel(f"P50 Derivative", color='red')
                ax_grad.tick_params(axis='x', labelcolor='red', colors='red')
                ax_grad.spines['top'].set_color('red')

                ax_prof.xaxis.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3, zorder=0)
                ax_prof.yaxis.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3, zorder=0)

                # Plotting Curves
                ax_prof.plot(row_p_min, heights, color='blue', linestyle='--', linewidth=1, zorder=98, label=f'P{minimum_percentil} / P{maximum_percentil}')
                ax_prof.plot(row_p_max, heights, color='blue', linestyle='--', linewidth=1, zorder=98)
                ax_prof.plot(row_median, heights, color='blue', linestyle='-', linewidth=2, zorder=98, label='P50')
                ax_grad.plot(gradient_profile, heights, color='red', linestyle='-', linewidth=1.5, zorder=99, label='P50 Derivative')

                # Legend Construction
                legend_label = (
                    f"----\n"
                    f"Max Contrast(|P{maximum_percentil} - P{minimum_percentil}|) = ({max_contrast_mid_h}px, {max_contrast_mid_val:.1f}%)\n"
                    f"Max Contrast(|P{minimum_percentil} - P0|) = ({max_contrast_shadow_h}px, {max_contrast_shadow_val:.1f}%)\n"
                    f"Max Contrast(|P100 - P{maximum_percentil}|) = ({max_contrast_highlight_h}px, {max_contrast_highlight_val:.1f}%)"
                )
                ax_prof.plot([], [], ' ', label=legend_label)

                # Legend Order
                lines_1, labels_1 = ax_prof.get_legend_handles_labels()
                lines_2, labels_2 = ax_grad.get_legend_handles_labels()
                final_lines = [lines_1[0], lines_1[1], lines_2[0], lines_1[2]]
                final_labels = [labels_1[0], labels_1[1], labels_2[0], labels_1[2]]

                leg1 = ax_grad.legend(final_lines, final_labels, loc='upper right',
                               fontsize='small', framealpha=1, facecolor='white', edgecolor='black', frameon=True)
                leg1.set_zorder(101)

                ax_prof.set_title(f"Vertical Profile Intensity", pad=35)
                plt.show()

                # =========================================================
                # OBJECT 2: HISTOGRAM
                # =========================================================
                # UPDATED v01.63: Set height to 6 to match Object 1.
                fig2 = plt.figure(figsize=(7.5, 6), layout='constrained')
                ax_hist = fig2.add_subplot(111)
                fig2.suptitle(f'Object 2 - Histogram Analysis: {file_name}', fontsize=14)

                ax_hist.set_xlabel("Pixel Intensity", color='black')
                ax_hist.tick_params(axis='x', colors='black')
                ax_hist.set_xlim(0, 255)
                ax_hist.set_ylabel("Pixel Count", color='black')
                ax_hist.tick_params(axis='y', colors='black')

                ax_cum = ax_hist.twinx()
                ax_cum.set_ylabel("Cumulative Percentage (%)", color='red')
                ax_cum.tick_params(axis='y', labelcolor='red', colors='red')
                ax_cum.spines['right'].set_color('red')
                ax_cum.set_yticks(np.arange(0, 101, 10))
                ax_cum.set_ylim(0, 105)

                n, bins, patches = ax_hist.hist(img_array.ravel(), bins=256, range=(0, 256),
                                              color='lightgray', edgecolor='none', density=False, label='Pixel Count', zorder=97)

                cdf = np.cumsum(n)
                cdf_normalized = (cdf / cdf[-1]) * 100
                bin_centers = (bins[:-1] + bins[1:]) / 2

                ax_cum.plot(bin_centers, cdf_normalized, color='red', linestyle='-', linewidth=2, label='Percentil', zorder=99)

                class_thresholds = [maximum_pixel_intensity_for_class_1, maximum_pixel_intensity_for_class_2,
                                    maximum_pixel_intensity_for_class_3, maximum_pixel_intensity_for_class_4]
                for th in class_thresholds:
                    ax_hist.axvline(th, color='blue', linestyle='-', linewidth=1.5, zorder=98)

                ax_hist.axvline(p50, color='darkgreen', linestyle='--', linewidth=2.5, zorder=98, label='P50')
                ax_hist.axvline(p_min, color='darkgreen', linestyle='--', linewidth=1, zorder=98, label=f'P{minimum_percentil} / P{maximum_percentil}')
                ax_hist.axvline(p_max, color='darkgreen', linestyle='--', linewidth=1, zorder=98)

                ax_hist.xaxis.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3, zorder=0)
                ax_cum.yaxis.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.3, zorder=0)

                legend_stats = (
                    f"----\n"
                    f"Contrast(|P{maximum_percentil} - P{minimum_percentil}|): {contrast_mid_global:.1f}%\n"
                    f"Contrast(|P{minimum_percentil} - P0|): {contrast_shadow_global:.1f}%\n"
                    f"Contrast(|P100 - P{maximum_percentil}|): {contrast_highlight_global:.1f}%"
                )
                ax_hist.plot([], [], ' ', label=legend_stats)

                lines_h, labels_h = ax_hist.get_legend_handles_labels()
                lines_c, labels_c = ax_cum.get_legend_handles_labels()

                final_lines_2 = [lines_h[0], lines_h[1], lines_h[2], lines_c[0], lines_h[3]]
                final_labels_2 = [labels_h[0], labels_h[1], labels_h[2], labels_c[0], labels_h[3]]

                leg2 = ax_cum.legend(final_lines_2, final_labels_2, loc='upper right',
                               framealpha=1, facecolor='white', edgecolor='black', fontsize='small', frameon=True)
                leg2.set_zorder(101)

                ax_hist.set_title("Pixel Intensity histogram (Count vs. Intensity)")

                plt.show()

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return results
