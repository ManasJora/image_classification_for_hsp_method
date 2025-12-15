
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
