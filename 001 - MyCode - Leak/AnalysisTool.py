""""
Program name: AnalysisTool.py
----------------------------------------------------------------------
Program elaborated to analyse previously acquired thermal images from a FLUKE camera.
----------------------------------------------------------------------
Description:
Python program used to analyse previously acquired thermal images from a FLUKE camera.
The program reads thermal images from a folder, processes them to detect leaks using explainability techniques, and saves the results to an output video file.
The program loops through the frames, applies various image processing techniques, and overlays the results on the original frames.
Some of the techniques used include entropy, variance, and region analysis to identify potential leak zones.

Environment:
  1. Python 3.12.7
  3. Matplotlib 3.7.0
  4. Numpy 1.26.4
  5. OpenCV 4.5.5
  6. MATLAB 2022a
  7. Seaborn 0.12.2
  8. scikit-image 0.22.0
  9. natsort 8.3.1
  10. pandas 2.1.3
  11. xlrd 2.0.1
  12. os 0.1.4

Version: 1.0
Modified on May 26th 2025
Author: Maria Rodrigues
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from skimage.morphology import remove_small_objects, disk, label
from skimage.filters.rank import entropy as sk_entropy
from skimage.measure import regionprops

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

# Set the style and context for clarity and boldness
sns.set_style("whitegrid", {"axes.grid": True, "axes.facecolor": "#FAFAFA"})
sns.set_context("notebook", font_scale=2.0)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.edgecolor'] = '#222222'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.alpha'] = 0.6
plt.rcParams['legend.frameon'] = False

# Use LaTeX 
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif', 'Times', 'Computer Modern', 'DejaVu Serif']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

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

# -------------- Folder Selection (unchanged, with brief comments) --------------

def fetch_folders(base_directory):
    """
    Build dictionaries mapping user menu choices to folder names for each depth.
    """
    first_level_folders = {
        '1': '000 - Testes a 1 metro',
        '2': '001 - Testes a 0,5 metro',
        'A': 'Analyze All'
    }
    second_level_folders = {
        '0': '000 - 0 Sem Aquecimento',
        '1': '001 - 0 Com Aquecimento',
        '2': '002 - 10 Sem Aquecimento',
        '3': '003 - 10 Com Aquecimento',
        '4': '004 - neg 10 Sem Aquecimento',
        '5': '005 - neg 10 Com Aquecimento',
        '6': '006 - 20 Sem Aquecimento',
        '7': '007 - 20 Com Aquecimento',
        '8': '008 - neg 20 Sem Aquecimento',
        '9': '009 - neg 20 Com Aquecimento',
        '10': '010 - 30 Sem Aquecimento',
        '11': '011 - 30 Com Aquecimento',
        '12': '012 - neg 30 Sem Aquecimento',
        '13': '013 - neg 30 Com Aquecimento',
        'A': 'Analyze All'
    }
    third_level_folders = {
        '0': '000 - Teste Sem Fuga',
        '1': '001 - Teste 1 mm',
        '2': '002 - Teste 0,5 mm',
        '3': '003 - Teste 0,25 mm',
        'A': 'Analyze All'
    }
    fourth_level_folders = {
        '1': '1',
        '2': '2',
        '3': '3',
        'A': 'Analyze All'
    }
    return first_level_folders, second_level_folders, third_level_folders, fourth_level_folders

def get_user_choice(prompt, options):
    """
    Presents options to user, validates selection.
    """
    print(prompt)
    for key, value in options.items():
        print(f"{key}: {value}")
    choice = input(f"Enter one of {list(options.keys())}: ").strip()
    while choice not in options:
        print("Invalid input. Please choose a valid option.")
        choice = input(f"Enter one of {list(options.keys())}: ").strip()
    return choice

def analyze_folder(folder_path):
    if folder_path is None:
        print("No folder path provided, aborting analysis.")
        return

    # process_videos(folder_path, second_choice)

    video_path = os.path.join(folder_path, "Video.avi")
    video_data_path = os.path.join(folder_path, "Video_data.avi")
    
    if os.path.isfile(video_path) and os.path.isfile(video_data_path):
        print(f"Opening videos in folder: {folder_path}")
        # process_videos(folder_path)
    else:
        print(f"Missing Video.avi or Video_data.avi in folder: {folder_path}")
        
def choose_path():
    base_directory = r"C:\\Users\\Maria Rodrigues\\Desktop\\Ficheiros Testes FLUKE"
    first_level_folders, second_level_folders, third_level_folders, fourth_level_folders = fetch_folders(base_directory)

    first_choice = get_user_choice("Choose first-level folder (or 'A' to analyze all):", first_level_folders)
    if first_choice == 'A':
        for k, folder in first_level_folders.items():
            if k != 'A':
                folder_path = os.path.join(base_directory, folder)
                analyze_folder(folder_path)
        return None, None

    second_choice = get_user_choice("Choose second-level folder (or 'A' to analyze all):", second_level_folders)
    if second_choice == 'A':
        for k, folder in second_level_folders.items():
            if k != 'A':
                folder_path = os.path.join(base_directory, first_level_folders[first_choice], folder)
                analyze_folder(folder_path)
        return None, None

    third_choice = get_user_choice("Choose third-level folder (or 'A' to analyze all):", third_level_folders)
    if third_choice == 'A':
        for k, folder in third_level_folders.items():
            if k != 'A':
                folder_path = os.path.join(base_directory, first_level_folders[first_choice], second_level_folders[second_choice], folder)
                analyze_folder(folder_path)
        return None, None

    fourth_choice = get_user_choice("Choose fourth-level folder (or 'A' to analyze all):", fourth_level_folders)
    if fourth_choice == 'A':
        for k, folder in fourth_level_folders.items():
            if k != 'A':
                folder_path = os.path.join(base_directory, first_level_folders[first_choice], second_level_folders[second_choice], third_level_folders[third_choice], folder)
                analyze_folder(folder_path)
        return None, None

    final_folder_path = os.path.join(
        base_directory,
        first_level_folders[first_choice],
        second_level_folders[second_choice],
        third_level_folders[third_choice],
        fourth_level_folders[fourth_choice]
    )

    if not os.path.exists(final_folder_path):
        print(f"Selected path does not exist: {final_folder_path}")
        return None, None

    return final_folder_path

# -------------- Data Reading & Conversion (docstrings and np.ptp fix) --------------

def is_csv_file(file_path):
    """Heuristically detects if a file is a CSV (vs. Excel)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            return first_line.strip().startswith('sep=') or ',' in first_line
    except Exception:
        return False


def read_data_file(file_path):
    """Reads thermal/CSV/Excel data, ensures shape, raises clear errors."""
    try:
        if is_csv_file(file_path):
            df = pd.read_csv(file_path, header=None, skiprows=4, nrows=480)
        else:
            ext = os.path.splitext(file_path)[1].lower()
            engine = 'xlrd' if ext == '.xls' else 'openpyxl'
            df = pd.read_excel(file_path, engine=engine, header=None, skiprows=4, nrows=480)
        data = df.iloc[:, 1:641].to_numpy()
        if data.shape != (480, 640):
            raise ValueError(f"Invalid shape in {file_path}: {data.shape}")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read data file {file_path}: {e}")

def temperature_to_grayscale(matrix, min_temp=10.0, max_temp=30.0):
    """
    Converts a thermal temperature matrix into a grayscale image.
    High temp = dark (so coldest pixels are visually highlighted).
    """
    denom = max_temp - min_temp
    if denom == 0:
        norm = np.zeros_like(matrix, dtype=np.uint8)
    else:
        norm = ((matrix - min_temp) / denom) * 255.0
        norm = np.nan_to_num(norm, nan=0.0, posinf=255.0, neginf=0.0)
    norm = np.clip(norm, 0, 255)
    inverted = 255 - norm  # invert so "cold spots" appear bright
    return inverted.astype(np.uint8)

def save_max_delta_matrix_excel(folder_path, output_excel_name):
    # Get all .xls files, natural order
    output_excel_path = os.path.join(folder_path, output_excel_name)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xls')]
    files = natsorted(files)
    if len(files) < 2:
        print("Need at least two files.")
        return

    # Initialize min and max matrices
    first_matrix = read_data_file(os.path.join(folder_path, files[0]))
    min_matrix = np.copy(first_matrix)
    max_matrix = np.copy(first_matrix)
    
    # Process each frame
    for fname in files[1:]:
        matrix = read_data_file(os.path.join(folder_path, fname))
        min_matrix = np.minimum(min_matrix, matrix)
        max_matrix = np.maximum(max_matrix, matrix)
    
    # Compute per-pixel max delta
    delta_matrix = max_matrix - min_matrix

    # Save as Excel file
    df_delta = pd.DataFrame(delta_matrix)
    df_delta.to_excel(output_excel_path, header=False, index=False)
    print(f"Max delta matrix saved to: {output_excel_path}")

def block_mean_2x2(matrix):
    """Downsamples a 2D matrix by averaging each non-overlapping 2x2 block."""
    # Ensure even dimensions
    h, w = matrix.shape
    h2, w2 = h // 2, w // 2
    matrix = matrix[:h2*2, :w2*2]  # trim if needed
    # Reshape and take mean
    return matrix.reshape(h2, 2, w2, 2).mean(axis=(1, 3))

def read_thermal_csv_xls(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=3, nrows=480)
    data = df.iloc[:, 1:641].to_numpy()
    return data

def save_max_delta_matrix_excel_downsampled(folder_path, output_excel_name="max_delta_matrix_2x2.xlsx"):
    # ... (use previous code to compute delta_matrix)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xls')]
    from natsort import natsorted
    files = natsorted(files)
    if len(files) < 2:
        print("Need at least two files.")
        return
    
    first_matrix = read_thermal_csv_xls(os.path.join(folder_path, files[0]))
    min_matrix = np.copy(first_matrix)
    max_matrix = np.copy(first_matrix)
    for fname in files[1:]:
        matrix = read_thermal_csv_xls(os.path.join(folder_path, fname))
        min_matrix = np.minimum(min_matrix, matrix)
        max_matrix = np.maximum(max_matrix, matrix)
    delta_matrix = max_matrix - min_matrix

    # Downsample by 2x2 blocks
    downsampled = block_mean_2x2(delta_matrix)

    # Save result
    output_excel_path = os.path.join(folder_path, output_excel_name)
    df_down = pd.DataFrame(downsampled)
    df_down.to_excel(output_excel_path, header=False, index=False)
    print(f"2x2 block-averaged max delta matrix saved to: {output_excel_path}")
    return downsampled

def apply_mask_to_thermal_video(folder_path, clean_mask, min_temp=10.0, max_temp=30.0):
    thermal_video_path = os.path.join(folder_path, "ThermalVideo.avi")
    masked_video_path = os.path.join(folder_path, "MaskedThermalVideo.avi")
    fps = 10
    if not os.path.isfile(thermal_video_path):
        print("ThermalVideo.avi not found, creating from XLS data...")
        create_thermal_video_from_xls(folder_path, thermal_video_path, min_temp, max_temp, fps=fps)
        if not os.path.isfile(thermal_video_path):
            raise RuntimeError("Could not create ThermalVideo.avi")
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

def find_largest_region(mask):
    """Returns the binary mask with only the largest connected region."""
    labels = label(mask.astype(np.uint8))
    if labels.max() == 0:
        return np.zeros_like(mask)
    largest = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    return largest

def area_from_metric_map(metric_map, mask_roi, min_area=10):
    # Find all connected regions in mask, get sum of metric in each, pick the largest sum/mean
    labels = label(mask_roi.astype(np.uint8))
    max_metric = -np.inf
    best_region = None
    for region in regionprops(labels):
        coords = tuple(zip(*region.coords))
        if len(coords[0]) < min_area:
            continue
        region_metric = metric_map[coords].mean()  # or sum()
        if region_metric > max_metric:
            max_metric = region_metric
            best_region = np.zeros_like(mask_roi, dtype=np.uint8)
            best_region[coords] = 1
    return best_region if best_region is not None else np.zeros_like(mask_roi, dtype=np.uint8)

def get_valid_frame_range(frame_count, fps=10, start_sec=3, end_sec=1):
    """
    Returns start and end frame indices to skip the first and last seconds,
    focusing analysis after the system stabilizes and before any leak is observable.
    This helps avoid artifacts or camera initialization effects.
    """
    start_frame = int(start_sec * fps)
    end_frame = frame_count - int(end_sec * fps)
    if end_frame <= start_frame:
        raise ValueError("Video too short to skip the specified segments.")
    return start_frame, end_frame

def analyze_and_mark_areas_combined(masked_video_path, mask, out_video_path, min_area=10, fps=10):
    """
    Analyze masked thermal video, overlaying on each frame:
      - Entropy region (static, red contour),
      - Variance region (per-frame, green contour),
      - Coldest pixel (per-frame, blue dot).
    For the final second, overlays 'Leak Zone' or 'True Leak Zone' and label.
    All overlays are on the same video.
    """

    import cv2
    import numpy as np
    import os
    from skimage.filters.rank import entropy as sk_entropy
    from skimage.morphology import disk
    from skimage.measure import label, regionprops

    def get_valid_frame_range(frame_count, fps=10, start_sec=3, end_sec=1):
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
    start_frame, end_frame = get_valid_frame_range(frame_count, fps=fps, start_sec=3, end_sec=1)
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

    
def make_side_by_side_video(folder_path, filenames, output_filename):
    readers = [cv2.VideoCapture(os.path.join(folder_path, f)) for f in filenames]
    width = int(readers[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(readers[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = readers[0].get(cv2.CAP_PROP_FPS)
    frame_count = int(readers[0].get(cv2.CAP_PROP_FRAME_COUNT))
    total_width = width * len(readers)
    out = cv2.VideoWriter(os.path.join(folder_path, output_filename),
                          cv2.VideoWriter_fourcc(*'XVID'), fps, (total_width, height), True)
    for i in range(frame_count):
        frames = []
        for reader in readers:
            ret, frame = reader.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)
        side_by_side = np.hstack(frames)
        out.write(side_by_side)
    for r in readers:
        r.release()
    out.release()
    print(f"Side-by-side video saved to: {os.path.join(folder_path, output_filename)}")

# -------------- Key Analysis Functions (with explainable comments) --------------

def create_thermal_video_from_xls(folder_path, video_path, min_temp=10.0, max_temp=30.0, fps=10):
    """
    Reads all .xls thermal frames in the folder, converts them to grayscale images, and writes a video.
    All steps (reading, normalization, frame stacking) are visible and reproducible.
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xls')]
    files = natsorted(files)
    if not files:
        print("No .xls files found.")
        return None, None

    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size, isColor=False)
    stack = []
    for file in files:
        full_path = os.path.join(folder_path, file)
        matrix = read_data_file(full_path)
        stack.append(matrix)
        grayscale = temperature_to_grayscale(matrix, min_temp, max_temp)
        out.write(grayscale)
    out.release()
    print(f"Thermal video created: {video_path}")
    return np.stack(stack, axis=0), files

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

def main():

    folder_path = choose_path()
    
    if not folder_path or not os.path.isdir(folder_path):
        print("No valid folder selected.")
        return
    
    save_max_delta_matrix_excel(folder_path, "max_delta_matrix.xlsx")
    
    downsampled = save_max_delta_matrix_excel_downsampled(folder_path)
    
    binary_mask_path = os.path.join(folder_path, "binary_mask.png")
    binary_mask = plot_binary_ok_mask(folder_path, downsampled, save_path=binary_mask_path)
    cleaned_mask = clean_binary_mask(folder_path, binary_mask, min_size=3)
    upsampled_mask = apply_mask_to_thermal_video(folder_path, cleaned_mask)
    
    masked_video_path = os.path.join(folder_path, "MaskedThermalVideo.avi")
    out_video_paths = {
        "entropy": os.path.join(folder_path, "MarkedMaskedThermalVideo_entropyArea.avi"),
        "variance": os.path.join(folder_path, "MarkedMaskedThermalVideo_varianceArea.avi"),
        "lightest": os.path.join(folder_path, "MarkedMaskedThermalVideo_lightest.avi"),
    }
    final_video_path = os.path.join(folder_path, "LeakZone.avi")
    analyze_and_mark_areas_combined(masked_video_path, cleaned_mask, final_video_path, min_area=10, fps=10)

    make_side_by_side_video(
        folder_path,
        [os.path.basename(out_video_paths[k]) for k in ("entropy", "variance", "lightest")],
        "comparison_areas_side_by_side.avi"
    )

if __name__ == "__main__":
    main()
