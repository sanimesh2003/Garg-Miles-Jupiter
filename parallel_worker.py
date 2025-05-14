# parallel_worker.py

import numpy as np
import cv2
import requests
import math
import os
import time # Added for sleep in retry logic
from astropy.io import fits
import traceback # Added for detailed error printing

# === Helper Functions (Copied from original notebook cells) ===

# --- From Cell 3 (Horizons API) ---
horizons_url = 'https://ssd.jpl.nasa.gov/api/horizons_file.api' # Define URL here

def create_input_content(dateobs, timeobs):
    # Assuming Jupiter ('599') and Earth Center ('500@399') are constants here
    # Match quantities requested in original cell 3 (9,20,23,24)
    return f"""
    !$$SOF
    COMMAND='599'
    OBJ_DATA='YES'
    MAKE_EPHEM='YES'
    TABLE_TYPE='OBSERVER'
    CENTER='500@399'
    TLIST='{dateobs} {timeobs}'
    QUANTITIES='9,20,23,24'
    CSV_FORMAT='YES'
    !$$EOF
    """
    # Changed to CSV_FORMAT='YES' for potentially easier parsing

def parse_horizons_csv_for_delta_and_sbrt(horizons_text):
    """Parses CSV formatted Horizons output for Delta and S-brt."""
    lines = horizons_text.splitlines()
    data_started = False
    delta = None
    s_brt = None

    for line in lines:
        if '$$SOE' in line:
            data_started = True
            continue
        if '$$EOE' in line:
            break
        if data_started and line.strip():
            try:
                # Example CSV columns based on QUANTITIES='9,20,23,24':
                # Date__(UT)__HR:MN:SC.fff, S-brt,     delta,    deldot, S-O-T, S-T-O
                # Check actual Horizons output for exact column order if CSV format varies
                fields = [f.strip() for f in line.split(',')]
                # Assuming S-brt is field index 1 and delta is field index 2 (0-based)
                # This might need adjustment based on the actual CSV output format!
                if len(fields) > 2:
                    # Check for empty strings before converting
                    if fields[1]: s_brt = float(fields[1])
                    if fields[2]: delta = float(fields[2])
                    if delta is not None and s_brt is not None:
                        return delta, s_brt # Return first valid row
            except (ValueError, IndexError) as e:
                # print(f"Warning: Could not parse line: {line} - {e}") # Optional debug
                continue # Skip lines that don't parse correctly
            except Exception as e_gen: # Catch other potential errors during parsing
                 print(f"Warning: General parsing error on line: {line} - {e_gen}")
                 continue

    # print(f"Warning: Could not find Delta or S-Brt in Horizons output.") # Optional debug if parsing fails
    return delta, s_brt # Return None, None if not found or only one was found

# --- Modified get_horizons_data with Retries ---
def get_horizons_data(dateobs, timeobs, retries=3, delay=3):
    """Fetches Horizons data with retries for temporary errors like 503."""
    content = create_input_content(dateobs, timeobs)
    last_exception = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                horizons_url,
                data={'format': 'text'}, # Request text format
                files={'input': ('input.txt', content)},
                timeout=45 # Increased timeout slightly
            )
            # Check for specific server errors like 503 first
            if resp.status_code == 503:
                 # Use worker pid for slightly clearer logs
                 print(f"Warning [Worker {os.getpid()}]: Horizons returned 503 (Attempt {attempt + 1}/{retries} for {dateobs} {timeobs}). Retrying in {delay}s...")
                 last_exception = requests.exceptions.HTTPError(f"503 Server Error after {retries} attempts")
                 time.sleep(delay) # Wait before retrying
                 continue # Go to next attempt

            # Raise HTTPError for other bad responses (4xx or 5xx)
            resp.raise_for_status()

            # Success, parse and return
            return parse_horizons_csv_for_delta_and_sbrt(resp.text)

        except requests.exceptions.RequestException as e:
            print(f"HORIZONS API ERROR [Worker {os.getpid()}] (Attempt {attempt + 1}/{retries} for {dateobs} {timeobs}): {e}")
            last_exception = e
            # Wait longer for general request errors before retrying, increase delay
            time.sleep(delay * (attempt + 1))

    # If all retries fail, print final warning and return None
    print(f"HORIZONS API ERROR [Worker {os.getpid()}]: Failed after {retries} attempts for {dateobs} {timeobs}. Last error: {last_exception}")
    return None, None


# --- From Cell 4 (Image Processing Utilities) ---
def elliptical_mask(shape, center, major_axis, minor_axis, angle_deg):
    (h, w) = shape
    # Ensure center coordinates are within bounds
    cx = np.clip(center[0], 0, w - 1)
    cy = np.clip(center[1], 0, h - 1)
    # Ensure axes are positive
    a = max(major_axis / 2.0, 1e-6) # Avoid zero division
    b = max(minor_axis / 2.0, 1e-6)

    y_grid, x_grid = np.ogrid[0:h, 0:w]
    theta = np.deg2rad(angle_deg)
    x_shifted = x_grid - cx
    y_shifted = y_grid - cy
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_prime = x_shifted * cos_t + y_shifted * sin_t
    y_prime = -x_shifted * sin_t + y_shifted * cos_t
    # Handle potential division by zero if a or b are extremely small
    # Ensure the result is boolean
    return ((x_prime**2) / (a**2) + (y_prime**2) / (b**2) <= 1.0).astype(bool)


def rotate_image_full(data, cx, cy, angle_deg):
    rows, cols = data.shape[:2] # Handle grayscale or color
    # Ensure cx, cy are within valid range
    cx_clipped = np.clip(cx, 0, cols - 1)
    cy_clipped = np.clip(cy, 0, rows - 1)
    M = cv2.getRotationMatrix2D((float(cx_clipped), float(cy_clipped)), float(angle_deg), 1.0)
    # Use INTER_LINEAR for smoother rotation, borderValue=0 for black background
    return cv2.warpAffine(data, M, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=0)

# === Worker Function (Moved from Notebook Cell 5) ===

def process_single_file(args):
    """
    Processes a single FITS file: reads data, calls Horizons, performs image processing,
    and returns calculated summaries and intermediate image arrays.

    Args:
        args (tuple): A tuple containing (index, file_detail_tuple).
                      index (int): The sequential index (1-based).
                      file_detail_tuple (tuple): Contains file_name, data, total_flux, etc.

    Returns:
        tuple: A tuple containing results for this file:
               (index, ellipse_summary_tuple, image_storage_dict, param_values_list,
                norm_8u_orig, norm_8u_rot, overlay_img, norm_8u_crop_for_npy)
               Returns None if processing fails.
    """
    i, file_detail_tuple = args
    # Unpack carefully, ensuring data is the NumPy array passed
    file_name, data, total_flux, date_obs, time_obs, telescope, instrument, exposure_time, t_filter = file_detail_tuple

    # Optional debug print
    # print(f"[Worker {os.getpid()}] Processing file #{i}: {file_name}")

    try:
        # --- Horizons API Call (uses retry logic now) ---
        dist_au, s_brt = get_horizons_data(date_obs, time_obs)

        # Initialize image variables
        norm_8u_orig = np.zeros((1, 1), dtype=np.uint8)
        norm_8u_rot = np.zeros((1, 1), dtype=np.uint8)
        overlay_img = np.zeros((1, 1, 3), dtype=np.uint8)
        norm_8u_crop = None
        cropped_data = np.array([])

        # --- Image Processing ---
        if data is not None and data.size > 1:
            # Use try-except blocks around potentially problematic calculations
            try:
                high_cut_orig = np.nanpercentile(data, 99) if np.any(np.isfinite(data)) else 0
                clipped_orig = np.clip(data, 0, high_cut_orig if np.isfinite(high_cut_orig) else np.inf)
                clipped_orig[~np.isfinite(clipped_orig)] = 0
                min_val, max_val = clipped_orig.min(), clipped_orig.max()

                if max_val > min_val:
                     norm_8u_orig = cv2.normalize(clipped_orig.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif max_val > 0:
                     norm_8u_orig = np.full(clipped_orig.shape, 255, dtype=np.uint8)
                else:
                     norm_8u_orig = np.zeros(clipped_orig.shape, dtype=np.uint8)
            except Exception as e_norm:
                 print(f"Warning: Error during initial normalization for {file_name}: {e_norm}")
                 norm_8u_orig = np.zeros((data.shape if data is not None else (1,1)), dtype=np.uint8) # Fallback
        else:
            data = np.zeros((1,1), dtype=np.float32) # Ensure data is an array

        # --- Rotation Calculation ---
        rotate_angle = 0.0
        rot_contours = []
        rotated_data = data
        rotated_flux = total_flux

        if data.ndim >= 2 and data.shape[0] > 0 and data.shape[1] > 0:
            cx, cy = (data.shape[1] / 2.0, data.shape[0] / 2.0)
            if norm_8u_orig.size > 1 and norm_8u_orig.ndim == 2:
                try: # Wrap thresholding and contour finding
                     tmp_thresh_val = 5
                     _, tmp_thresh = cv2.threshold(norm_8u_orig, tmp_thresh_val, 255, cv2.THRESH_BINARY)
                     if tmp_thresh is not None and tmp_thresh.dtype == np.uint8:
                          tmp_contours, _ = cv2.findContours(tmp_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                          if len(tmp_contours) > 0:
                               largest_tmp_contour = max(tmp_contours, key=cv2.contourArea)
                               if len(largest_tmp_contour) >= 5:
                                    (temp_cx, temp_cy), (w, h), angle = cv2.fitEllipse(largest_tmp_contour)
                                    if w > 0 and h > 0:
                                         if h > w: angle += 90
                                    rotate_angle = angle
                except cv2.error as e_cv2:
                     print(f"Warning: cv2 error during initial contour/ellipse for {file_name}: {e_cv2}")
                except Exception as e_contour:
                     print(f"Warning: Error during initial contour/ellipse for {file_name}: {e_contour}")

            # --- Actual Rotation and Normalization ---
            try:
                rotated_data = rotate_image_full(data, cx, cy, rotate_angle)
                rotated_flux = float(np.sum(rotated_data))

                high_cut_rot = np.nanpercentile(rotated_data, 99) if np.any(np.isfinite(rotated_data)) else 0
                clipped_rot = np.clip(rotated_data, 0, high_cut_rot if np.isfinite(high_cut_rot) else np.inf)
                clipped_rot[~np.isfinite(clipped_rot)] = 0
                min_val_rot, max_val_rot = clipped_rot.min(), clipped_rot.max()

                if max_val_rot > min_val_rot:
                     norm_8u_rot = cv2.normalize(clipped_rot.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif max_val_rot > 0:
                     norm_8u_rot = np.full(clipped_rot.shape, 255, dtype=np.uint8)
                else:
                     norm_8u_rot = np.zeros(clipped_rot.shape, dtype=np.uint8)
            except Exception as e_rot:
                 print(f"Warning: Error during rotation/normalization for {file_name}: {e_rot}")
                 rotated_data = data # Fallback
                 rotated_flux = total_flux
                 norm_8u_rot = norm_8u_orig # Fallback

            # --- Overlay Image Prep and Final Contours ---
            if norm_8u_rot.ndim == 2:
                overlay_img = cv2.cvtColor(norm_8u_rot, cv2.COLOR_GRAY2BGR)
            else: # Assume already BGR or handle error?
                overlay_img = np.zeros((1, 1, 3), dtype=np.uint8) # Fallback

            try: # Wrap final threshold and contour
                 threshold_value = 5
                 if norm_8u_rot.size > 1 and norm_8u_rot.dtype == np.uint8:
                      _, thr_rot = cv2.threshold(norm_8u_rot, threshold_value, 255, cv2.THRESH_BINARY)
                      if thr_rot is not None: rot_contours, _ = cv2.findContours(thr_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                      else: rot_contours = []
                 else: rot_contours = []
            except Exception as e_cont2:
                 print(f"Warning: Error during final contour finding for {file_name}: {e_cont2}")
                 rot_contours = []
        else: # Initial data was invalid
            rotated_data = data
            rotated_flux = total_flux
            norm_8u_rot = norm_8u_orig
            overlay_img = np.zeros((1, 1, 3), dtype=np.uint8)
            rot_contours = []

        # --- Calculations based on rotated data contours ---
        ellipse_flux = 0.0
        cropped_flux = 0.0
        flux_avg = 0.0
        ef_err = 0.0
        fa_err = 0.0

        if len(rot_contours) > 0:
            try:
                rot_largest_contour = max(rot_contours, key=cv2.contourArea)
                if len(rot_largest_contour) >= 5:
                    (rcx, rcy), (rw, rh), rangle = cv2.fitEllipse(rot_largest_contour)
                    if rw > 0 and rh > 0:
                        ellipse_major = max(rw, rh)
                        ellipse_minor = min(rw, rh)
                        ellipse_angle = rangle

                        # Check rotated_data validity before masking
                        if rotated_data is not None and rotated_data.ndim == 2 and rotated_data.shape[0] > 0 and rotated_data.shape[1] > 0:
                            mask_ell = elliptical_mask(rotated_data.shape, (rcx, rcy), ellipse_major, ellipse_minor, ellipse_angle)
                            if mask_ell.shape == rotated_data.shape:
                                 ellipse_flux = float(np.sum(rotated_data[mask_ell]))

                        if overlay_img.size > 1:
                             center_pt = (int(round(rcx)), int(round(rcy)))
                             axes_len = (int(round(ellipse_minor)), int(round(ellipse_major)))
                             if axes_len[0] > 0 and axes_len[1] > 0:
                                 cv2.ellipse(overlay_img, (center_pt, axes_len, ellipse_angle), (0, 255, 0), 2)

                # Bounding Rect and Cropping
                rx, ry, rW, rH = cv2.boundingRect(rot_largest_contour)
                # Check rotated_data validity before cropping
                if rW > 0 and rH > 0 and rotated_data is not None and rotated_data.ndim == 2 and \
                   ry >= 0 and rx >= 0 and ry+rH <= rotated_data.shape[0] and rx+rW <= rotated_data.shape[1]:
                    cropped_data = rotated_data[ry:ry + rH, rx:rx + rW].copy()
                    cropped_flux = float(np.sum(cropped_data))
                    if overlay_img.size > 1:
                         cv2.rectangle(overlay_img, (rx, ry), (rx + rW, ry + rH), (255, 0, 0), 2)
                else:
                    cropped_data = np.array([]) # Ensure it's empty if crop failed
            except cv2.error as e_cv2_final:
                 print(f"Warning: cv2 error during final contour/ellipse/rect for {file_name}: {e_cv2_final}")
                 cropped_data = np.array([])
            except Exception as e_calc:
                 print(f"Warning: Error during final calculations for {file_name}: {e_calc}")
                 cropped_data = np.array([])


            # Final calculations
            if ellipse_flux > 0:
                ef_err = np.sqrt(ellipse_flux)
                valid_exptime = False
                try:
                    exp_time_float = float(exposure_time)
                    if exp_time_float > 0: valid_exptime = True
                except: pass

                if valid_exptime and dist_au is not None and dist_au != 0:
                    flux_avg = ellipse_flux / (exp_time_float * dist_au)
                    fa_err = ef_err / (exp_time_float * dist_au)
                else:
                    flux_avg = np.inf
                    fa_err = np.inf


        # --- Prepare return values ---
        ellipse_summary_tuple = (i, file_name, s_brt, ellipse_flux, dist_au, ef_err, flux_avg, fa_err, date_obs, time_obs)

        param_values_list = [
            ("File #", str(i)), ("File Name", file_name), ("DATE-OBS", date_obs), ("TIME-OBS", time_obs),
            ("Telescope", telescope), ("Instrument", instrument), ("Filter", t_filter),
            ("surface brightness (Jupiter)", f"{s_brt:.3f}" if s_brt is not None else "N/A"),
            ("Total Flux", f"{total_flux:.2f}"), ("Flux (Rotated)", f"{rotated_flux:.2f}"),
            ("Ellipse Flux", f"{ellipse_flux:.2f}"),
            ("Cropped Flux", f"{cropped_flux:.2f}" if cropped_data.size > 0 else "N/A"),
            ("Exposure Time", f"{exposure_time}"), ("Delta (AU)", f"{dist_au:.5f}" if dist_au is not None else "N/A"),
            ("Flux Average", f"{flux_avg:.2f}" if flux_avg != 0.0 and np.isfinite(flux_avg) else "N/A"),
            ("Ellipse Flux Err", f"{ef_err:.2f}"), ("Flux Avg Err", f"{fa_err:.2f}" if np.isfinite(fa_err) else "N/A")
        ]

        norm_8u_crop_for_npy = None
        if cropped_data.size > 1:
            try: # Wrap final normalization
                high_cut_crop = np.nanpercentile(cropped_data, 99) if np.any(np.isfinite(cropped_data)) else 0
                clipped_crop = np.clip(cropped_data, 0, high_cut_crop if np.isfinite(high_cut_crop) else np.inf)
                clipped_crop[~np.isfinite(clipped_crop)] = 0
                min_val_crop, max_val_crop = clipped_crop.min(), clipped_crop.max()
                if max_val_crop > min_val_crop:
                     norm_8u_crop_for_npy = cv2.normalize(clipped_crop.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif max_val_crop > 0:
                     norm_8u_crop_for_npy = np.full(clipped_crop.shape, 255, dtype=np.uint8)
                else:
                     norm_8u_crop_for_npy = np.zeros(clipped_crop.shape, dtype=np.uint8)
            except Exception as e_norm_crop:
                 print(f"Warning: Error normalizing cropped image for {file_name}: {e_norm_crop}")
                 norm_8u_crop_for_npy = None # Ensure None if error

        image_storage_dict = {
            "index": i, "file_name": file_name, "date_obs": date_obs, "time_obs": time_obs,
            "orig_image": norm_8u_orig if norm_8u_orig.size > 1 else None,
            "rotated_image": norm_8u_rot if norm_8u_rot.size > 1 else None,
            "contour_image": overlay_img if overlay_img.size > 1 else None,
            "cropped_image": norm_8u_crop_for_npy # Use the variable specifically for storage/npy
        }

        # Return results successfully
        return (i, ellipse_summary_tuple, image_storage_dict, param_values_list,
                norm_8u_orig if norm_8u_orig.size > 1 else None,
                norm_8u_rot if norm_8u_rot.size > 1 else None,
                overlay_img if overlay_img.size > 1 else None,
                norm_8u_crop_for_npy)

    except Exception as e_main:
        # Log error for this specific file
        print(f"!!! MAJOR ERROR processing file #{i} ({file_name}) in worker {os.getpid()}: {e_main}")
        traceback.print_exc() # Print full traceback from worker
        return None # Indicate failure
