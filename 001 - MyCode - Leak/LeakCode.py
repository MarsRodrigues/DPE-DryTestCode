"""
Program name: LeakCode.py
----------------------------------------------------------------------
Program to acquire thermal images from a FLUKE camera through MATLAB, in real time, a
----------------------------------------------------------------------
Description:
Python program to get thermal images from a FLUKE camera through MATLAB engine, via serial communication.
The program captures thermal images from the camera and saves them to a video file.
Each raw thermal frame is also saved as a separate Excel file (Data_1.xls, Data_2.xls, ...).
The program uses the MATLAB engine to connect to the camera and get the thermal images.
The program displays the thermal images in a window using OpenCV.

Environment:
  1. Python 3.12.7
  2. pandas 2.2.2
  3. Numpy 1.26.4
  4. OpenCV 4.5.5
  5. MATLAB 2022a
  6. time 3.10.0

Version: 1.1
Modified on May 27th 2025
Author: Maria Rodrigues
"""
import time
import matlab.engine
import numpy as np
import cv2
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from skimage.morphology import remove_small_objects, disk, label
from skimage.filters.rank import entropy as sk_entropy
from skimage.measure import regionprops

def acqmain():
    
    def block_mean_2x2(matrix):
        """Downsamples a 2D matrix by averaging each non-overlapping 2x2 block."""
        h, w = matrix.shape
        h2, w2 = h // 2, w // 2
        matrix = matrix[:h2*2, :w2*2]
        return matrix.reshape(h2, 2, w2, 2).mean(axis=(1, 3))

    def get_max_delta_matrix_2x2_from_buffer(thermal_frames):
        """
        Compute the per-pixel max delta matrix from a list of 2D arrays (thermal_frames),
        downsample with 2x2 block mean, and return as a numpy array.
        """
        if len(thermal_frames) < 2:
            raise ValueError("Need at least two frames.")

        buffer_matrix = np.stack(thermal_frames, axis=0)
        min_matrix = np.min(buffer_matrix, axis=0)
        max_matrix = np.max(buffer_matrix, axis=0)
        delta_matrix = max_matrix - min_matrix
        downsampled = block_mean_2x2(delta_matrix)
        return downsampled

    output_directory = r"C:\Users\Maria Rodrigues\Desktop\ThermalAnalysis"
    video_filename = "ThermalVideo.avi"
    video_path = os.path.join(output_directory, video_filename)

    frame_width = 640
    frame_height = 480
    fps = 20
    duration_seconds = 20  # Capture for 20 seconds

    os.makedirs(output_directory, exist_ok=True)
    start_time = time.time()
    
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    eng.ToolkitFunctions.LoadAssemblies(nargout=0)
    print("Assemblies Loaded.")
    CameraSerial = eng.ToolkitFunctions.DiscoverDevices()
    CameraSerial = CameraSerial[0]

    if eng.ToolkitFunctions.SelectDevice(CameraSerial) == True:
        print(CameraSerial + ': Camera Successfully Opened.')

    cameraStream = eng.ToolkitFunctions.StartStream()
    cameratime = time.time() - start_time
    print(f"Connection to camera took: {cameratime:.2f} seconds.")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    thermal_frames = []
    ct = time.time()

    try:
        start_time = time.time()
        while time.time() - ct < duration_seconds:
            IRImage = eng.ToolkitFunctions.GetData('Celsius')
            IRImage_np = np.array(IRImage._data).reshape(IRImage.size, order='F')

            # Ensure 2D
            if IRImage_np.ndim == 3:
                IRImage_np = IRImage_np[:, :, 0]

            thermal_frames.append(IRImage_np.copy())

            # Visualization for video/output (normalize and colorize)
            norm_img = cv2.normalize(IRImage_np, None, 0, 255, cv2.NORM_MINMAX)
            norm_img = np.uint8(norm_img)
            color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            color_img_resized = cv2.resize(color_img, (frame_width, frame_height))

            out.write(color_img_resized)
            # cv2.imshow("Thermal Image", color_img_resized)

        image_time = time.time() - start_time
        print(f" Frames captured in {image_time:.2f} seconds.", end='\r')
        
    finally:
        eng.ToolkitFunctions.StopStream(nargout=0)
        eng.quit()
        out.release()
        # cv2.destroyAllWindows()
        print("Video acquisition finished.")

        start_time = time.time()
        # Compute and store downsampled max delta matrix (Celsius)
        print("Computing downsampled max delta matrix (in memory)...")
        downsampled_delta = get_max_delta_matrix_2x2_from_buffer(thermal_frames)
        np.save("downsampled_delta.npy", downsampled_delta)
        print("Downsampled 2x2 max delta matrix computed.")
        analytic_time = time.time() - start_time
        print(f"Total time for matrix analysis: {analytic_time:.2f} seconds.")
        
    return downsampled_delta

"""
Main function for Leak Analysis.
This function will read the downsampled delta matrix and perform analysis.
It will also handle video processing and overlaying results.
"""

# --- Seaborn and Matplotlib Style ---
# Cohesive color palette (mainly blue/green/red for clarity)
blue_palette = sns.color_palette("Blues", 8)
red_palette = sns.color_palette("Reds", 8)
green_palette = sns.color_palette("Greens", 8)
grey_palette = sns.color_palette("Greys", 8)

# Custom palette for multi-line plots
custom_palette = [
    blue_palette[6],  # Main line
    green_palette[5], # Secondary
    red_palette[5],   # Tertiary
    grey_palette[6],  # Reference/baseline
]

# --- Utility for cohesive plots ---
def make_cohesive_axes(ax):
    """Standardize axes for plots: bold, grid, and minimal spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_axisbelow(True)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')

# -------------- HELPER FUNCTIONS --------------

def legend_overlay(frame):
    legend = [
        ((255, 255, 255),  "Red Cross: Variance"),
        ((255, 255, 255),  "Blue Box: ROI"),
        ((255, 255, 255),  "Blue Dot: Coldest pixel"),
        ((255, 255, 255),  "Green Dot: Max Entropy"),
        ((255, 255, 255),  "Red: Temperature Delta"),
        ((255, 255, 255),  "Yellow Arrows: Motion Vector"),
        ((255, 255, 255),  "White Edges: ROI Texture Edges"),
    ]
    overlay = frame.copy()
    # Soft background rectangle
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    for i, (color, desc) in enumerate(legend):
        y = 10 + 20 * (i + 1)
        # Draw marker (circle or line as color key)
        if "dot" in desc:
            cv2.circle(frame, (32, y), 9, color, -1)
        elif "cross" in desc:
            cv2.line(frame, (24, y - 8), (40, y + 8), color, 3)
            cv2.line(frame, (24, y + 8), (40, y - 8), color, 3)
        elif "box" in desc or "edges" in desc:
            cv2.rectangle(frame, (22, y - 10), (42, y + 10), color, 2)
        elif "arrows" in desc:
            cv2.arrowedLine(frame, (24, y), (40, y), color, 3, tipLength=0.6)
        else:
            cv2.circle(frame, (9, y), 3, color, -1)
        # Put text (always white, left aligned, no color)
        cv2.putText(frame, desc, (14, y + 1), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    return frame


def frame_explanation(frame, text):
    """Overlay a single-line explanation at the bottom for human-understandable feature attribution."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h-38), (w, h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, text, (14, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# -------------- Data Reading & Conversion (docstrings and np.ptp fix) --------------

def apply_mask_to_thermal_video(folder_path, clean_mask, min_temp=10.0, max_temp=30.0):
    thermal_video_path = os.path.join(folder_path, "ThermalVideo.avi")
    masked_video_path = os.path.join(folder_path, "MaskedThermalVideo.avi")
    fps = 10
    if not os.path.isfile(thermal_video_path):
        print("ThermalVideo.avi not found, creating from XLS data...")

    cap = cv2.VideoCapture(thermal_video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read a frame from the thermal video")
    frame_shape = frame.shape[:2]  # (H, W)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    mask_resized = cv2.resize(
        clean_mask.astype(np.uint8),
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(masked_video_path, fourcc, fps, (frame_shape[1], frame_shape[0]), isColor=True)
    light_red = np.array([180, 180, 255], dtype=np.uint8)
    alpha = 0.3
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Applying mask to {frame_count} frames...")
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            gray = frame if len(frame.shape) == 2 else frame[:, :, 0]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = np.stack([gray] * 3, axis=-1)
        out_frame = gray_bgr.copy()
        mask_inv = ~mask_resized
        out_frame[mask_inv] = (alpha * light_red + (1 - alpha) * gray_bgr[mask_inv]).astype(np.uint8)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out_frame, contours, -1, (255, 255, 255), 1)
        out.write(out_frame)
    cap.release()
    out.release()
    print(f"Masked thermal video saved to {masked_video_path}")
    return mask_resized

def analyze_and_mark_areas_combined(masked_video_path, mask, out_video_path, min_area=10, fps=24):
    """
    Analyze masked thermal video, overlaying on each frame:
    - Entropy region (static, red contour),
    - Variance region (per-frame, green contour),
    - Coldest pixel (per-frame, blue dot).
    For the final second, overlays 'Leak Zone' or 'True Leak Zone' and label.
    All overlays are on the same video.
    """

    def get_valid_frame_range(frame_count, fps=24, start_sec=3, end_sec=1):
        start_frame = int(start_sec * fps)
        end_frame = frame_count - int(end_sec * fps)
        if end_frame <= start_frame:
            raise ValueError("Video too short to skip the specified segments.")
        return start_frame, end_frame

    def find_largest_region(mask):
        labels = label(mask.astype(np.uint8))
        if labels.max() == 0:
            return np.zeros_like(mask)
        largest = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
        return largest

    def area_from_metric_map(metric_map, mask_roi, min_area=10):
        labels = label(mask_roi.astype(np.uint8))
        max_metric = -np.inf
        best_region = None
        for region in regionprops(labels):
            coords = tuple(zip(*region.coords))
            if len(coords[0]) < min_area:
                continue
            region_metric = metric_map[coords].mean()
            if region_metric > max_metric:
                max_metric = region_metric
                best_region = np.zeros_like(mask_roi, dtype=np.uint8)
                best_region[coords] = 1
        return best_region if best_region is not None else np.zeros_like(mask_roi, dtype=np.uint8)

    # --- Load all valid frames ---
    cap = cv2.VideoCapture(masked_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, end_frame = get_valid_frame_range(frame_count, fps=fps, start_sec=0, end_sec=3)
    frames = []
    color_frames = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i < start_frame or i >= end_frame:
            continue
        color_frames.append(frame.copy())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(gray)
    cap.release()
    stack = np.stack(frames)
    color_frames = np.stack(color_frames)
    height, width = stack.shape[1:]
    valid_frame_count = stack.shape[0]

    # --- Prepare mask ---
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask_roi = mask > 0.5

    # --- Output video writer ---
    # Ensure output path is a string, properly formatted
    if not isinstance(out_video_path, str):
        raise ValueError(f"Output video path is not a string: {out_video_path}")
    out_video_path = os.path.abspath(out_video_path)
    out_dir = os.path.dirname(out_video_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving combined analysis video to: {out_video_path}")

    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), True)
    if not out.isOpened():
        raise IOError(f"Could not open video writer with path: {out_video_path}")

    # --- Static entropy delta region (from first/last frame) ---
    ent_first = sk_entropy(stack[0], disk(5))
    ent_last = sk_entropy(stack[-1], disk(5))
    ent_delta = ent_last - ent_first
    entropy_area_mask = area_from_metric_map(ent_delta, mask_roi, min_area=min_area)
    entropy_area_mask = find_largest_region(entropy_area_mask)

    # --- Per-frame variance region, and coldest pixel ---
    variance_regions = []
    lightest_points = []
    matches_with_entropy = 0
    for i in range(valid_frame_count):
        # Variance up to frame i
        var_map = np.var(stack[:i+1], axis=0)
        var_area = area_from_metric_map(var_map, mask_roi, min_area=min_area)
        var_area = find_largest_region(var_area)
        variance_regions.append(var_area)
        # Lightest
        masked_frame = stack[i].astype(np.float32)
        masked_frame[~mask_roi] = -np.inf
        y, x = np.unravel_index(np.argmax(masked_frame), masked_frame.shape)
        lightest_points.append((y, x))
        # Check if variance region matches entropy region
        if np.array_equal(var_area, entropy_area_mask) and var_area.sum() > 0:
            matches_with_entropy += 1

    # --- Rule: If >3 frames had variance region == entropy region, that's the leak zone ---
    if matches_with_entropy > 3:
        final_zone = entropy_area_mask
        label_text = "True Leak Zone"
    else:
        # Usual: most frequent variance region not containing coldest pixel
        freq_map = np.zeros((height, width), dtype=np.int32)
        for region_mask in variance_regions:
            freq_map += region_mask
        freq_values = np.unique(freq_map)[::-1]
        candidate_masks = []
        for val in freq_values:
            if val < 1:
                continue
            candidate = (freq_map == val).astype(np.uint8)
            candidate = find_largest_region(candidate)
            if candidate.sum() == 0:
                continue
            candidate_masks.append(candidate)
        # Disqualify if any frame's lightest point is in region
        final_zone = None
        for region in candidate_masks:
            contained = any(region[y, x] for (y, x) in lightest_points)
            if not contained:
                final_zone = region
                break
        if final_zone is None:
            final_zone = np.zeros_like(mask_roi, dtype=np.uint8)
        label_text = "Leak Zone"

    # Prepare label placement for leak zone
    cnts, _ = cv2.findContours(final_zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts and len(cnts[0]) > 0:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        label_x = min(x + w + 15, width - 180)
        label_y = max(y + h // 2, 40)
    else:
        label_x = 30
        label_y = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    # --- Overlay all analyses on each frame and write to video ---
    for i in range(valid_frame_count):
        frame_out = color_frames[i].copy()

        # Entropy region (static, red)
        cnts_entropy, _ = cv2.findContours(entropy_area_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_out, cnts_entropy, -1, (0, 0, 255), 2)

        # Variance region (green, per-frame)
        cnts_var, _ = cv2.findContours(variance_regions[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame_out, cnts_var, -1, (0, 255, 0), 2)

        # Coldest (lightest) pixel (blue dot)
        y_l, x_l = lightest_points[i]
        cv2.circle(frame_out, (x_l, y_l), 6, (255,255,255), 2)
        cv2.circle(frame_out, (x_l, y_l), 4, (255,0,0), -1)

        # In final second, add leak zone and label
        if i >= valid_frame_count - fps and final_zone.sum() > 0:
            cv2.drawContours(frame_out, cnts, -1, (0, 0, 255), 3)
            cv2.putText(frame_out, label_text, (label_x+2, label_y+2), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
            cv2.putText(frame_out, label_text, (label_x, label_y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

        out.write(frame_out)

    out.release()
    print(f"Combined analysis video saved (frames {start_frame}–{end_frame - 1} of {frame_count}):\n  {out_video_path}")
    if label_text == "True Leak Zone":
        print(">3 frames matched entropy and variance regions: labeled 'True Leak Zone'.")
    else:
        print("Leak zone is most frequent variance region not explained by coldest pixel.")

# -------------- Key Analysis Functions (with explainable comments) --------------

def plot_binary_ok_mask(folder_path, matrix, save_path=None):
    """
    Creates and displays (and optionally saves) a binary mask
    where 1 means 0.5 < value < 0.8 ("OK" points), 0 elsewhere.
    """
    ok_mask = (matrix > 0.5) & (matrix < 0.8)
    binary_mask = ok_mask.astype(np.uint8)
    plt.figure(figsize=(12, 8))
    plt.imshow(binary_mask, cmap='gray', interpolation='nearest', aspect='auto')
    plt.title("Binary Mask: OK Points (Delta Temperature Between 0.5 and 0.8)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"Binary mask saved to {save_path}")
        
    return binary_mask

def clean_binary_mask(folder_path, binary_mask, min_size=5):
    """
    Removes small isolated regions (noisy pixels) from the binary mask.
    min_size: Minimum number of pixels a region must have to be kept.
    """
    # Ensure mask is boolean for skimage
    cleaned = remove_small_objects(binary_mask.astype(bool), min_size=min_size)
    return cleaned.astype(np.uint8)

# --- Utility: Title overlay ---

def put_title(image, text):
    scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    shadow = (40, 40, 40)
    org = (10, 30)
    cv2.putText(image, text, (org[0]+1, org[1]+1), cv2.FONT_HERSHEY_SIMPLEX, scale, shadow, thickness+2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return image

# --- Main ---

def main(downsampled_delta):

    folder_path = r"C:\Users\Maria Rodrigues\Desktop\ThermalAnalysis"
        
    binary_mask_path = os.path.join(folder_path, "binary_mask.png")
    binary_mask = plot_binary_ok_mask(folder_path, downsampled_delta, save_path=binary_mask_path)
    cleaned_mask = clean_binary_mask(folder_path, binary_mask, min_size=3)
    upsampled_mask = apply_mask_to_thermal_video(folder_path, cleaned_mask)
    
    masked_video_path = os.path.join(folder_path, "MaskedThermalVideo.avi")
    final_video_path = os.path.join(folder_path, "LeakZone.avi")
    analyze_and_mark_areas_combined(masked_video_path, cleaned_mask, final_video_path, min_area=10, fps=24)
    print("Analysis complete. Results saved in the specified folder.")

def main():
    start_time = time.time()
    
    # Initialize the acquisition module
    downsampled_delta = acqmain()
    acquisition_time = time.time() - start_time
        
    # Initialize the leak analysis module
    analysis_time = time.time()
    main(downsampled_delta)
    analysis_time = time.time() - analysis_time
    print(f"Leak Analysis completed in {analysis_time:.2f} seconds.")
    
    # Print the results
    print("Leak Analysis Completed.")
    print(f"Total time taken: {acquisition_time + analysis_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()
