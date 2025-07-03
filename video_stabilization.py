import numpy as np
import cv2
import time

def apply_nms(keypoints, radius):
    """
    if not keypoints:
        return []
    
    # Sort keypoints by response strength (higher is better)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    filtered_keypoints = []

    for kp in keypoints:
        keep = True
        for fkp in filtered_keypoints:
            dist = np.linalg.norm(np.array(kp.pt) - np.array(fkp.pt))
            if dist < radius:
                keep = False
                break
        if keep:
            filtered_keypoints.append(kp)

    return filtered_keypoints
    """
    if not keypoints:
        return []

    pts = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sorted_idx = np.argsort(-responses)
    pts = pts[sorted_idx]

    grid_size = radius
    x_coords = pts[:, 0] // grid_size
    y_coords = pts[:, 1] // grid_size
    grid = {}
    keep_mask = np.ones(len(pts), dtype=bool)

    for i in range(len(pts)):
        x = x_coords[i]
        y = y_coords[i]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (x + dx, y + dy)
                if cell in grid:
                    for j in grid[cell]:
                        if np.linalg.norm(pts[i] - pts[j]) < radius:
                            keep_mask[i] = False
                            break
                    if not keep_mask[i]:
                        break
            if not keep_mask[i]:
                break

        if keep_mask[i]:
            if (x, y) not in grid:
                grid[(x, y)] = []
            grid[(x, y)].append(i)

    return [keypoints[sorted_idx[i]] for i in np.where(keep_mask)[0]]


def feature_detect(prev_gray, curr_gray):

    scale_factor = 2

    prev_gray = cv2.resize(prev_gray, (prev_gray.shape[1] // scale_factor, prev_gray.shape[0] // scale_factor))
    curr_gray = cv2.resize(curr_gray, (curr_gray.shape[1] // scale_factor, curr_gray.shape[0] // scale_factor))

    lk_params = dict( winSize  = (30,30),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    orb = cv2.ORB_create(nfeatures=100, scaleFactor=1.01, nlevels=15, WTA_K=4, edgeThreshold=5, patchSize=15)
    prev_kp = orb.detect(prev_gray, None)
    print("Previous Keypoints:", len(prev_kp))
    #print(prev_kp)
    # Apply NMS to filter keypoints
    prev_kp = apply_nms(prev_kp, radius = 10)
    #print("After NMS:", len(prev_kp))
    prev_points = np.array([kp.pt for kp in prev_kp], dtype=np.float32).reshape(-1, 1, 2)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params) 
    good_new = curr_pts[status == 1]
    good_old = prev_points[status == 1]

    good_old *= scale_factor
    good_new *= scale_factor

    #print("good_new:", len(good_new))
    #print("good_old:", len(good_old))

    #good_new_kp = [cv2.KeyPoint(x, y, 1) for x, y in good_new.reshape(-1, 2)]
    #good_old_kp = [cv2.KeyPoint(x, y, 1) for x, y in good_old.reshape(-1, 2)]

    displacements = np.linalg.norm(good_new - good_old, axis=1)
    #print("Displacemnts:", displacements)
    displacement_threshold = 100  # Adjust based on your requirement
    filtered_indices = displacements < displacement_threshold
    #print("No. of filtered_indices:", filtered_indices)
    
    # Apply filtering
    filtered_good_old = good_old[filtered_indices]
    filtered_good_new = good_new[filtered_indices]
    good_old_kp = [cv2.KeyPoint(x, y, 1) for x, y in filtered_good_old.reshape(-1, 2)]
    good_new_kp = [cv2.KeyPoint(x, y, 1) for x, y in filtered_good_new.reshape(-1, 2)]

    #print("good_old_kp:", len(good_old_kp))
    #print("good_new_kp:", len(good_new_kp))
    
    return good_old, good_new, good_old_kp, good_new_kp

def motion_estimate(good_old, good_new, motion_history, average_window):
    dx = np.mean(good_new[:, 0] - good_old[:, 0])
    dy = np.mean(good_new[:, 1] - good_old[:, 1])
    #print("dx dy:", dx, dy)
    #visualize the matching points in the frame
    # Store motion history for smoothing
    motion_history.append((dx, dy))
    if len(motion_history) > average_window:
        motion_history.pop(0)
    
    # Compute smoothed dx and dy
    avg_dx = np.mean([m[0] for m in motion_history])
    avg_dy = np.mean([m[1] for m in motion_history])
    return dx, dy, avg_dx, avg_dy

def motion_smooth(dx, dy, avg_dx, avg_dy, curr_frame):
    frame_height, frame_width = curr_frame.shape[:2]
    h_avg_dx = dx - 0.1*avg_dx
    h_avg_dy = dy - 0.1*avg_dy
    # Apply motion correction
    transform_matrix = np.array([[1, 0, -h_avg_dx], [0, 1, -h_avg_dy]], dtype=np.float32)

    stabilized_frame = cv2.warpAffine(curr_frame, transform_matrix, (frame_width, frame_height))

    mask = (stabilized_frame == 0).all(axis=2).astype(np.uint8)  # Identify black pixels (all channels zero)

    # Fill black borders using the previous frame
    filled_frame = stabilized_frame.copy()
    filled_frame[mask == 1] = curr_frame[mask == 1]

    # Optional: Blend the filled areas for smoother transition
    alpha = 0.1  # Weight for stabilized frame
    blended_frame = cv2.addWeighted(stabilized_frame, alpha, filled_frame, 1 - alpha, 0)

    result_frame = cv2.hconcat([curr_frame, blended_frame])
    return h_avg_dx, h_avg_dy, stabilized_frame, result_frame




def stabilization(prev_frame, curr_frame):
    motion_history = []
    average_window = 20

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    st = time.time()

    good_old, good_new,good_old_kp, good_new_kp = feature_detect(prev_gray, curr_gray)
    
    dx, dy,avg_dx, avg_dy = motion_estimate(good_old, good_new, motion_history, average_window)

    h_avg_dx, h_avg_dy, stabilized_frame, result_frame = motion_smooth(dx, dy, avg_dx, avg_dy, curr_frame)

    matches = [cv2.DMatch(i, i, 0) for i in range(len(good_new_kp))]  # create a list of matches
    #print("Matches:", len(matches))
    result = cv2.drawMatches(prev_frame, good_old_kp, curr_frame, good_new_kp, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    if (result_frame.shape[1] > 1500):
        result_frame = cv2.resize(result_frame, (1500, 780))
    #print(result_frame.shape)
    #out.write(result_frame)
    #cv2.imshow('Stabilized_frame', result_frame)
    prev_frame = stabilized_frame.copy()
    et = time.time()
    fps = round(1 / (et - st))
    print("FPS:", fps)
    #dx_values.append(dx)
    return result_frame
    #dy_values.append(dy)
    #avg_dx_v.append(avg_dx)
    #avg_dy_v.append(avg_dy)
    #h_avg_dx_.append(h_avg_dx)
    #h_avg_dy_.append(h_avg_dy)

