import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from tqdm import tqdm

# ====================================================================
# USER CONFIGURATION SECTION
# --------------------------------------------------------------------
# 1. PATHS: Please update these three paths before running.
#
# Recommended Project Structure:
# /project_root_directory
#   /data (Contains all .xlsx files, e.g., AIA glc3_Train_1.xlsx)
#   /results (Output folder for results and figures)
#   /celegans_bodybend_analysis.py (This script)
# ====================================================================
# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(BASE_DIR, "results")
# Use os.path.join to ensure the FIGURE_FILE_NAME is correctly built
FIGURE_FILE_NAME_FOR_PLOTTING = "AIA glc3_Train_1.xlsx"
FRAME_RATE = 10
# ====================================================================
# 2. ANALYSIS PARAMETERS: Modify only if you understand the effect
# ====================================================================
SMOOTH_WINDOW = 5
DIST_WINDOW = 21
PEAK_PARAMS = {
    'prominence': 0.0045,
    'width': (1.0, 35.0)
}

# === Experiment Info Parsing Function ===
def parse_experiment_info(file_path):
    file_name = os.path.basename(file_path)

    return {
        'file_name': file_name,
        'file_path': file_path
    }

def moving_average(data, window_size):
    """Moving Average Filter"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def analyze_movement(file_path):
    """Core analysis function, returns data for plotting and statistical features."""
    try:
        data = pd.read_excel(file_path)
        x = data['x'].values
        y = data['y'].values

        # === Step 0: Calculate Average Velocity using raw trajectory data ===
        dx = np.diff(x)
        dy = np.diff(y)
        total_distance = np.sqrt(dx ** 2 + dy ** 2).sum()
        total_time = (len(data) - 1) / FRAME_RATE
        avg_velocity = total_distance / total_time

        # === Step 1: Trajectory Smoothing (Baseline Calculation) ===
        x_smooth_raw = moving_average(x, SMOOTH_WINDOW)  # 长度 N - SMOOTH_WINDOW + 1
        y_smooth_raw = moving_average(y, SMOOTH_WINDOW)

        # === Step 2: Calculate Offset Distance (Waving Amplitude) ===
        trim_start_x_raw = SMOOTH_WINDOW // 2
        trim_end_x_raw = trim_start_x_raw + len(x_smooth_raw)
        distances = np.sqrt((x[trim_start_x_raw: trim_end_x_raw] - x_smooth_raw) ** 2 +
                            (y[trim_start_x_raw: trim_end_x_raw] - y_smooth_raw) ** 2)  # 长度 N - SMOOTH_WINDOW + 1

        # === Step 3: Distance Signal Smoothing (Further Filtering) ===
        dist_smooth = savgol_filter(distances, DIST_WINDOW, 3)  # N - SMOOTH_WINDOW - DIST_WINDOW + 2

        # === Step 4: Unify Timeline and Trim Data (CORRECTED LOGIC) ===

        trim_start_raw = SMOOTH_WINDOW // 2 + DIST_WINDOW // 2
        trim_end_raw = trim_start_raw + len(dist_smooth)
        x_trimmed = x[trim_start_raw:trim_end_raw]
        y_trimmed = y[trim_start_raw:trim_end_raw]

        trim_start_smooth_raw = DIST_WINDOW // 2
        trim_end_smooth_raw = trim_start_smooth_raw + len(dist_smooth)
        x_smooth_trimmed = x_smooth_raw[trim_start_smooth_raw: trim_end_smooth_raw]
        y_smooth_trimmed = y_smooth_raw[trim_start_smooth_raw: trim_end_smooth_raw]

        distances_trimmed = distances[trim_start_smooth_raw: trim_end_smooth_raw]

        # === Step 5: Extrema Detection ===
        peaks_max, _ = find_peaks(dist_smooth, **PEAK_PARAMS)
        peaks_min, _ = find_peaks(-dist_smooth, **PEAK_PARAMS)

        # === Step 6: Period/Amplitude Calculation & Statistics (FIXED align_extrema) ===
        def align_extrema(max_t, min_t):
            """Aligns maxima and minima into a single list (FIXED)"""
            pairs = []
            i = j = 0
            while i < len(max_t) and j < len(min_t):
                if max_t[i] < min_t[j]:
                    pairs.append(('max', max_t[i]))
                    i += 1
                else:
                    pairs.append(('min', min_t[j]))
                    j += 1
            # Append remaining extrema (Crucial FIX)
            while i < len(max_t):
                pairs.append(('max', max_t[i]))
                i += 1
            while j < len(min_t):
                pairs.append(('min', min_t[j]))
                j += 1
            return pairs

        ext_pairs = align_extrema(peaks_max, peaks_min)
        periods = []
        amplitudes = []

        for k in range(len(ext_pairs) - 1):
            if ext_pairs[k][0] == 'max' and ext_pairs[k + 1][0] == 'min':
                amp = dist_smooth[ext_pairs[k][1]] - dist_smooth[ext_pairs[k + 1][1]]
                amplitudes.append(amp)

            if ext_pairs[k][0] == 'max' and ext_pairs[k + 1][0] == 'min':
                t_max = ext_pairs[k][1] / FRAME_RATE
                t_min = ext_pairs[k + 1][1] / FRAME_RATE
                periods.append(2 * (t_min - t_max))

        # Statistical Feature Calculation
        stats = {
            'avg_velocity': avg_velocity,
            'period_mean': np.nanmean(periods) if periods else np.nan,
            'period_median': np.nanmedian(periods) if periods else np.nan,
            'period_std': np.nanstd(periods) if periods else np.nan,
            'period_q1': np.nanquantile(periods, 0.25) if periods else np.nan,
            'period_q3': np.nanquantile(periods, 0.75) if periods else np.nan,
            'amp_mean': np.nanmean(amplitudes) if amplitudes else np.nan,
            'amp_median': np.nanmedian(amplitudes) if amplitudes else np.nan,
            'amp_std': np.nanstd(amplitudes) if amplitudes else np.nan,
            'amp_q1': np.nanquantile(amplitudes, 0.25) if amplitudes else np.nan,
            'amp_q3': np.nanquantile(amplitudes, 0.75) if amplitudes else np.nan,
            'num_cycles': len(periods)
        }

        return {
            'x': x_trimmed,
            'y': y_trimmed,
            'x_smooth': x_smooth_trimmed,
            'y_smooth': y_smooth_trimmed,
            'distances': distances_trimmed,
            'dist_smooth': dist_smooth,
            'peaks_max': peaks_max,
            'peaks_min': peaks_min,
            'ext_pairs': ext_pairs,
            'periods_list': periods,
            'amplitudes_list': amplitudes,
            **parse_experiment_info(file_path),
            **stats
        }

    except Exception as e:
        print(f"Processing failed: {file_path}\nError type: {type(e).__name__}\nDetails: {str(e)}")
        return None

# === Batch Processing Main Program ===
def batch_analyze(root_dir, output_path):
    """Batch processes all Excel files in the root_dir and saves results."""
    # 1. Collect all files
    all_files = []
    # os.walk will traverse subdirectories, but the break stops it after the root level.
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(('.xlsx', '.xls')):
                all_files.append(os.path.join(root, f))
        break

    print(f"Found {len(all_files)} data files in the root directory.")

    # 2. Processing with progress bar
    results = []
    error_files = []

    for file_path in tqdm(all_files, desc="Analysis Progress"):
        result = analyze_movement(file_path)
        if result:
            stats_and_info = {k: v for k, v in result.items() if isinstance(v, (str, float, int))}
            results.append(stats_and_info)
        else:
            error_files.append(file_path)

    # 3. Result handling
    if not results:
        print("Warning: No files were processed successfully!")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 4. Save results
    try:
        # Check if output_path is a directory or if it needs to be created
        if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
            final_output_path = os.path.join(output_path, 'analysis_results.xlsx')

            os.makedirs(output_path, exist_ok=True)
            print(f"Saving results to directory, using default filename: {final_output_path}")
        else:
            final_output_path = output_path
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

        with pd.ExcelWriter(final_output_path) as writer:
            # Full results
            df.to_excel(writer, sheet_name='Analysis Results', index=False)

            # Error log
            if error_files:
                pd.DataFrame({'Error File': error_files}).to_excel(
                    writer, sheet_name='Failed Files', index=False)

    except Exception as e:
        print(f"Error saving results to {final_output_path if 'final_output_path' in locals() else output_path}: {e}")

    print(f"\nProcessing complete! Successfully analyzed {len(results)} files, failed {len(error_files)}")
    return df


# === Plotting Functions ===

def plot_movement_analysis(figure_data, start_frame, end_frame, save_path=None):
    """
    Plots the analysis figures based on the given data and frame range.
    [Figure A: Trajectory, Figure B: Distance Function, Figure C: Period & Amplitude Vis]
    """
    if figure_data is None:
        print("No plotting data available.")
        return

    # Trim data to the specified frame range
    x_seg = figure_data['x'][start_frame:end_frame]
    y_seg = figure_data['y'][start_frame:end_frame]
    x_smooth_seg = figure_data['x_smooth'][start_frame:end_frame]
    y_smooth_seg = figure_data['y_smooth'][start_frame:end_frame]
    dist_smooth_seg = figure_data['dist_smooth'][start_frame:end_frame]

    # Adjust extrema indices based on the trimmed data
    peaks_max_seg = figure_data['peaks_max'][
                        (figure_data['peaks_max'] >= start_frame) & (
                                    figure_data['peaks_max'] < end_frame)] - start_frame
    peaks_min_seg = figure_data['peaks_min'][
                        (figure_data['peaks_min'] >= start_frame) & (
                                    figure_data['peaks_min'] < end_frame)] - start_frame

    # --- Global Font Size Settings ---
    plt.rcParams.update({'font.size': 8})

    # Color customization
    color_gray = np.array([111, 111, 111]) / 255
    color_blue = np.array([154, 201, 219]) / 255
    color_amplitude = np.array([40, 120, 181]) / 255
    color_red = np.array([255, 136, 132]) / 255
    color_period = np.array([200, 36, 35]) / 255

    # --- Use subplot_mosaic for flexible layout ---
    fig, axes = plt.subplot_mosaic([['ax1', '.'],
                                    ['ax2', 'ax2'],
                                    ['ax3', 'ax3']], figsize=(8, 11))

    fig.suptitle(f"Movement Analysis ({figure_data['file_name']})\nFrames: {start_frame}-{end_frame}")

    # Plot 1: Actual and Smoothed Trajectory (A) - FIXED: Added labels to eliminate UserWarning
    ax1 = axes['ax1']
    ax1.plot(x_smooth_seg, y_smooth_seg, color=color_red, linestyle='-', linewidth=1,
             label='Smoothed Trajectory') # <--- FIXED
    ax1.plot(x_seg, y_seg, color='k', marker='o', linestyle='-', linewidth=1, markersize=2, alpha=0.5,
             label='Actual Trajectory') # <--- FIXED

    ax1.set_title('A: Actual Movement Curve and the Baseline')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    # ax1.legend() will now find labels and the UserWarning will be gone
    ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    ax1.set_aspect('equal', adjustable='box')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(0.05, 0.95, 'A', transform=ax1.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')


    # Plot 2: Distance Function (B)
    ax2 = axes['ax2']
    ax2.plot(dist_smooth_seg, color=color_gray, linestyle='-', linewidth=1, label='Smoothed Distance')
    ax2.set_title('B: Distance Function')
    ax2.set_xlabel(f'Frame Index')
    ax2.set_ylabel('Distance from Smoothed Trajectory')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.045, 1), frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.text(0.05, 0.95, 'B', transform=ax2.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')

    # Plot 3: Visualization of Period and Amplitude (C)
    ax3 = axes['ax3']
    ax3.plot(dist_smooth_seg, color=color_gray, linestyle='-', linewidth=1)
    ax3.plot(peaks_max_seg, dist_smooth_seg[peaks_max_seg], color=color_red, marker='o', linestyle='', markersize=5,
             label='Maxima')
    ax3.plot(peaks_min_seg, dist_smooth_seg[peaks_min_seg], color=color_blue, marker='o', linestyle='', markersize=5,
             label='Minima')

    # --- Add Period and Amplitude Visualization ---
    target_amplitude_index = 1
    target_period_index = 2

    # Re-align extrema for the segment to find pairs
    extrema_seg = []
    i, j = 0, 0
    while i < len(peaks_max_seg) or j < len(peaks_min_seg):
        if i < len(peaks_max_seg) and (j == len(peaks_min_seg) or peaks_max_seg[i] < peaks_min_seg[j]):
            extrema_seg.append(('max', peaks_max_seg[i]))
            i += 1
        elif j < len(peaks_min_seg):
            extrema_seg.append(('min', peaks_min_seg[j]))
            j += 1
        else:
            break

    amplitude_count = 0
    period_count = 0

    for k in range(len(extrema_seg) - 1):
        if extrema_seg[k][0] == 'max' and extrema_seg[k + 1][0] == 'min':
            if amplitude_count == target_amplitude_index:
                amp_peak_idx = extrema_seg[k][1]
                amp_valley_idx = extrema_seg[k + 1][1]

                # Draw vertical line for Amplitude
                ax3.vlines(x=amp_peak_idx,
                           ymin=dist_smooth_seg[amp_valley_idx],
                           ymax=dist_smooth_seg[amp_peak_idx],
                           color=color_amplitude, linestyle='-', linewidth=2)

                # Add Amplitude helper horizontal line
                ax3.hlines(y=dist_smooth_seg[amp_valley_idx],
                           xmin=amp_valley_idx,
                           xmax=amp_peak_idx,
                           color=color_amplitude, linestyle='--', linewidth=1)

                # Adjust Amplitude text position
                text_x_pos = amp_peak_idx + (amp_peak_idx - amp_valley_idx) * 0.1
                ax3.text(text_x_pos,
                         (dist_smooth_seg[amp_valley_idx] + dist_smooth_seg[amp_peak_idx]) / 2.2,
                         'Amplitude', ha='right', va='center', color=color_amplitude)
            amplitude_count += 1

        # Period (Max to next Max)
        if k < len(extrema_seg) - 2 and extrema_seg[k][0] == 'max' and extrema_seg[k + 2][0] == 'max':
            if period_count == target_period_index:
                period_start_idx = extrema_seg[k][1]
                period_end_idx = extrema_seg[k + 2][1]

                # Draw horizontal line for Period
                ax3.hlines(y=dist_smooth_seg[period_end_idx],
                           xmin=period_start_idx,
                           xmax=period_end_idx,
                           color=color_period, linestyle='-', linewidth=2)

                # Add Period helper vertical line
                ax3.vlines(x=period_start_idx,
                           ymin=dist_smooth_seg[period_start_idx],
                           ymax=dist_smooth_seg[period_end_idx],
                           color=color_period, linestyle='--', linewidth=1)

                ax3.text((period_start_idx + period_end_idx) / 2,
                         dist_smooth_seg[period_start_idx],
                         'Period', ha='center', va='bottom', color=color_period)
            period_count += 1

    ax3.set_title('Distance Function with Period and Amplitude')
    ax3.set_xlabel(f'Frame Index')
    ax3.set_ylabel('Distance from Smoothed Trajectory')
    ax3.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.text(0.05, 0.95, 'C', transform=ax3.transAxes, va='top', ha='left', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        try:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving figure to {save_path}: {e}")
    else:
        plt.show()


def plot_zoomed_in_trajectory(figure_data, start_frame, end_frame, save_path=None):
    """
    Plots a zoomed-in version of the raw and baseline trajectory.
    """
    if figure_data is None:
        print("No plotting data available.")
        return

    # Trim data to the specified frame range
    x_seg = figure_data['x'][start_frame:end_frame]
    y_seg = figure_data['y'][start_frame:end_frame]
    x_smooth_seg = figure_data['x_smooth'][start_frame:end_frame]
    y_smooth_seg = figure_data['y_smooth'][start_frame:end_frame]

    # Set color
    color_red = np.array([255, 136, 132]) / 255

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(x_smooth_seg, y_smooth_seg, color=color_red, linestyle='-', linewidth=1,
            label='Smoothed Trajectory (Baseline)')
    ax.plot(x_seg, y_seg, color='k', marker='o', linestyle='-', linewidth=1, markersize=2, alpha=0.5,
            label='Actual Trajectory')

    ax.set_title(f"Zoomed-in Trajectory ({figure_data['file_name']})\nFrames: {start_frame}-{end_frame}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(frameon=False)

    # Use ax.axis('equal') to ensure X and Y axis scales are the same
    ax.axis('equal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        try:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)  # 关闭图片以释放内存
        except Exception as e:
            print(f"Error saving zoom figure to {save_path}: {e}")
    else:
        plt.show()  # 如果没有提供保存路径，则显示图片

def main_plot_segmented_analysis(file_path_for_figure, segment_size_pairs=10):
    """
    Main function to process a single file, segment, and plot.
    """
    print(f"\n--- Starting Plotting Analysis for: {os.path.basename(file_path_for_figure)} ---")
    figure_data = analyze_movement(file_path_for_figure)

    # Check if analysis was successful and if the required list keys exist
    if figure_data is None or 'ext_pairs' not in figure_data:
        print("Analysis failed or missing data for plotting.")
        return

    # Get the aligned extrema pairs on the unified timeline
    ext_pairs = figure_data['ext_pairs']

    if not ext_pairs:
        print("No extrema pairs detected, plotting the full trajectory.")
        plot_movement_analysis(figure_data, 0, len(figure_data['x']))
        return

    plot_dir = os.path.join(OUTPUT_PATH, 'figures')
    os.makedirs(plot_dir, exist_ok=True)
    base_name = os.path.splitext(figure_data['file_name'])[0]

    # Calculate segments based on the number of extrema pairs
    num_pairs = len(ext_pairs)
    num_segments = (num_pairs + segment_size_pairs - 1) // segment_size_pairs

    print(
        f"Total {num_pairs} extrema pairs detected. Data will be segmented into {num_segments} sections for visualization.")

    # --- Segmented Plotting ---
    for i in range(num_segments):
        start_pair_index = i * segment_size_pairs
        end_pair_index = min((i + 1) * segment_size_pairs, num_pairs)

        start_frame = ext_pairs[start_pair_index][1]
        end_frame = ext_pairs[end_pair_index - 1][1]
        if end_pair_index < num_pairs:
            # If it's not the last segment, use the starting index of the next segment as the end frame
            end_frame = ext_pairs[end_pair_index][1]

        end_frame = min(end_frame, len(figure_data['x']))

        print(f"Plotting segment {i + 1}/{num_segments}, frame range: [{start_frame}, {end_frame}]")

        # 构造分段图表的保存路径
        save_file = os.path.join(plot_dir, f"{base_name}_segment_{i + 1}_{start_frame}-{end_frame}.png")
        plot_movement_analysis(figure_data, start_frame, end_frame, save_path=save_file)  # <-- 传递保存路径

    # --- Plot Zoomed-in Segment ---
    print("\n--- Segmented plotting completed ---")

    zoom_start = 180
    zoom_end = 210
    print(f"Plotting default zoomed-in segment, frame range: [{zoom_start}, {zoom_end}]")

    # 构造局部放大图的保存路径
    zoom_save_file = os.path.join(plot_dir, f"{base_name}_zoom_{zoom_start}-{zoom_end}.png")
    plot_zoomed_in_trajectory(figure_data, zoom_start, zoom_end, save_path=zoom_save_file)

if __name__ == '__main__':
    # Get the full path to the figure file now that ROOT_DIR is defined
    FIGURE_FILE_PATH = os.path.join(ROOT_DIR, FIGURE_FILE_NAME_FOR_PLOTTING)

    # ----------------------------------------------------------------
    # --- Phase 1: Batch Analysis ---
    # ----------------------------------------------------------------
    print("--- Phase 1: Starting Batch Analysis ---")
    df_results = batch_analyze(ROOT_DIR, OUTPUT_PATH)

    if not df_results.empty:
        print(f"Batch analysis completed. Results saved to: {OUTPUT_PATH}")
        print("\nAnalysis Results (File Name and Key Metrics Only):")
        print(df_results[['file_name', 'avg_velocity', 'period_mean', 'amp_mean']].head())
    else:
        print("Batch analysis did not return any valid data.")

    # ----------------------------------------------------------------
    # --- Phase 2: Single File Plotting ---
    # ----------------------------------------------------------------
    if os.path.exists(FIGURE_FILE_PATH):
        print(f"\n--- Phase 2: Starting Plotting for: {os.path.basename(FIGURE_FILE_PATH)} ---")
        main_plot_segmented_analysis(FIGURE_FILE_PATH, segment_size_pairs=10)
    else:
        print(f"\nWarning: Plotting skipped. Figure file not found at: {FIGURE_FILE_PATH}\n"
              f"Please ensure '{os.path.basename(FIGURE_FILE_NAME_FOR_PLOTTING)}' exists in '{ROOT_DIR}'.")