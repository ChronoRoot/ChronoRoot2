""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

Optimized for efficiency by processing expensive operations in ROIs 
and pre-allocating morphological kernels.
"""

from skimage.morphology import skeletonize
import cv2
import numpy as np

# --- PRE-ALLOCATED KERNELS (Global Scope) ---
# Moving these out of the functions prevents re-allocation on every loop/call.

# Kernels for trim()
T_KERNELS = [
    np.array([[-1, 1, -1], [1, 1, 1], [0, 0, 0]]),
    np.array([[-1, 1, 0], [1, 1, 0], [-1, 1, 0]]),
    np.array([[0, 0, 0], [1, 1, 1], [-1, 1, -1]]),
    np.array([[0, 1, -1], [0, 1, 1], [0, 1, -1]]),
    np.array([[1, -1, -1], [1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, 1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [1, 1, -1]]),
    np.array([[-1, 1, 1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, 1]]),
    np.array([[-1, 1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]]),
    np.array([[-1, 1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, 1, -1]])
]

# Kernels for endPoints() - Strict background
EP_KERNELS = [
    np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, 1, -1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, 1], [-1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [1, 1, -1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, 1], [-1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]]),
    np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
]

# Perfectly symmetrical relaxed kernels for pruning
PRUNE_KERNELS = [
    # Straight directions (Relaxed corners using 0)
    np.array([[-1, -1, -1], [-1,  1, -1], [ 0,  1,  0]]), # UP
    np.array([[ 0,  1,  0], [-1,  1, -1], [-1, -1, -1]]), # DOWN
    np.array([[-1, -1,  0], [-1,  1,  1], [-1, -1,  0]]), # LEFT
    np.array([[ 0, -1, -1], [ 1,  1, -1], [ 0, -1, -1]]), # RIGHT
    
    # Diagonal directions (Strict corners)
    np.array([[-1, -1, -1], [-1,  1, -1], [ 1, -1, -1]]), # NE tip (Branch SW)
    np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1,  1]]), # NW tip (Branch SE)
    np.array([[ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]]), # SE tip (Branch NW)
    np.array([[-1, -1,  1], [-1,  1, -1], [-1, -1, -1]])  # SW tip (Branch NE)
]

# Kernels for branchedPoints()
X_KERNELS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
]

T_BRANCH_KERNELS = [
    np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]]),
    np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]),
    np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]]),
    np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]]),
    np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]]),
    np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]]),
    np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]]),
    np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
]

Y_KERNELS = [
    np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]]),
    np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]]),
    np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]]),
    np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]]),
    np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
]
# Add rotated versions to Y_KERNELS
_Y3 = np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
_Y4 = np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
Y_KERNELS.append(np.rot90(_Y3))
Y_KERNELS.append(np.rot90(_Y4))
Y_KERNELS.append(np.rot90(np.rot90(_Y3)))


def get_roi_bounding_box(mask, padding=5):
    """
    Helper to find the bounding box of non-zero pixels.
    Returns slices for cropping and offsets for coordinate restoration.
    """
    points = cv2.findNonZero(mask)
    if points is None:
        return None, None
    
    x, y, w, h = cv2.boundingRect(points)
    
    # Apply padding while staying within image bounds
    h_img, w_img = mask.shape
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(w_img, x + w + padding)
    y_end = min(h_img, y + h + padding)
    
    roi_slice = (slice(y_start, y_end), slice(x_start, x_end))
    offset = (x_start, y_start)
    
    return roi_slice, offset

def calculate_bbox_iou(box1, box2):
    """Calculates Intersection over Union for two bounding boxes (x_min, y_min, x_max, y_max)."""
    if box1 is None or box2 is None:
        return 0.0
        
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def extract_root_segmentation(segmentation_path, roi_bounds, current_root_base, fixed_seed_position, previous_bbox=None):
    # Load segmentation and crop to region of interest
    multi_class_mask = cv2.imread(segmentation_path, 0)[roi_bounds[0]:roi_bounds[1], roi_bounds[2]:roi_bounds[3]]
    
    # Extract hypocotyl length
    hypocotyl_skeleton, hypocotyl_length = extract_hypocotyl_length(multi_class_mask)
    
    # Remove anything above the original seed point (not part of root)
    multi_class_mask[0:fixed_seed_position[1], :] = 0

    # Combine segmentation classes (1, 2 are root classes)
    binary_mask = (multi_class_mask == 1) + (multi_class_mask == 2)
    binary_mask = np.array(binary_mask, dtype='uint8') * 255
    
    # Clean up mask with morphological operations 
    morph_kernel_size = 5
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    
    # Find all connected components
    connected_components, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for component in connected_components:
        area = cv2.contourArea(component)
        if area < 30:  # Skip tiny components
            continue
            
        # Get bounding box for the component
        x, y, w, h = cv2.boundingRect(component)
        comp_bbox = (x, y, x + w, y + h)
        
        # Calculate distance to root base
        dist_raw = cv2.pointPolygonTest(component, (int(current_root_base[0]), int(current_root_base[1])), True)
        distance_to_root_base = np.abs(dist_raw)
        
        # If the point is inside the contour, distance is effectively 0
        if dist_raw >= 0:
            distance_to_root_base = 0.0 
            
        # Calculate IoU with previous frame
        iou = calculate_bbox_iou(previous_bbox, comp_bbox) if previous_bbox is not None else 1.0
        
        candidates.append({
            'component': component,
            'area': area,
            'distance': distance_to_root_base,
            'iou': iou,
            'bbox': comp_bbox
        })

    if not candidates:
        return binary_mask, hypocotyl_skeleton, hypocotyl_length, False, binary_mask, None

    best_candidate = None

    if previous_bbox is None:
        # INITIALIZATION PHASE: No previous BBox. Pick the largest component near the root base.
        candidates.sort(key=lambda x: x['area'], reverse=True)
        for cand in candidates:
            if cand['distance'] < 100:
                best_candidate = cand
                break
    else:
        # TRACKING PHASE: Filter by IoU > 0, then select the biggest IoU. If tie, select closest to root base.
        valid_candidates = [c for c in candidates if c['iou'] > 0]
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: (x['iou'], -x['area']), reverse=True)
            best_candidate = valid_candidates[0]

    # If we found a valid component based on our rules, process it
    if best_candidate is not None:
        component = best_candidate['component']
        
        component_mask = np.zeros(binary_mask.shape, np.uint8)
        cv2.drawContours(component_mask, [component], -1, 255, -1)
        filtered_mask = cv2.bitwise_and(component_mask, binary_mask.copy())
        
        temp_mc_mask = np.zeros_like(multi_class_mask)
        temp_mc_mask[multi_class_mask == 1] = 2
        temp_mc_mask[multi_class_mask == 2] = 1
        
        mc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_temp = cv2.dilate(temp_mc_mask, mc_kernel)
        
        dilated_mc_mask = np.zeros_like(multi_class_mask)
        dilated_mc_mask[dilated_temp == 2] = 1
        dilated_mc_mask[dilated_temp == 1] = 2
        
        mc_filtered_mask = dilated_mc_mask * (filtered_mask > 0).astype(np.uint8)

        return filtered_mask, hypocotyl_skeleton, hypocotyl_length, True, mc_filtered_mask, best_candidate['bbox']
    
    # No component passed the filters
    return binary_mask, hypocotyl_skeleton, hypocotyl_length, False, binary_mask, None

def extract_hypocotyl_length(multi_class_mask):
    # Filter Mask for Hypocotyls (Class 4)
    binary_mask = (multi_class_mask == 4) 
    binary_mask = np.array(binary_mask, dtype='uint8') * 255
    
    morph_kernel_size = 5
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    
    morph_kernel_size = 3
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    binary_mask = cv2.erode(binary_mask, morph_kernel)
    binary_mask = cv2.dilate(binary_mask, morph_kernel)
    
    connected_components, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Pre-calculate Area and BoundingRect for valid components only
    components_data = []
    for comp in connected_components:
        area = cv2.contourArea(comp)
        if area > 30: 
            rect = cv2.boundingRect(comp)
            components_data.append((area, comp, rect))
    
    if not components_data:
        return np.zeros_like(binary_mask), 0
    
    # Sort: Largest area first
    components_data.sort(key=lambda x: x[0], reverse=True)
    
    component_mask = np.zeros(binary_mask.shape, np.uint8)
    accepted_rects = [] 
    GAP_THRESHOLD = 100 
    
    for i, (_, component, rect) in enumerate(components_data):
        # Always keep the largest component (Index 0)
        if i == 0:
            cv2.drawContours(component_mask, [component], -1, 255, -1)
            accepted_rects.append(rect)
            continue
        
        # Efficient Proximity Check 
        # We check this new component against all previously accepted ones.
        x1, y1, w1, h1 = rect
        is_close = False
        
        for (rx, ry, rw, rh) in accepted_rects:
            # Calculate gap distance using simple arithmetic (no square roots)
            # Horizontal gap
            if x1 + w1 < rx: h_gap = rx - (x1 + w1)
            elif rx + rw < x1: h_gap = x1 - (rx + rw)
            else: h_gap = 0
            
            # Vertical gap
            if y1 + h1 < ry: v_gap = ry - (y1 + h1)
            elif ry + rh < y1: v_gap = y1 - (ry + rh)
            else: v_gap = 0
            
            # If gap is small in both dimensions, it's a match
            if max(h_gap, v_gap) < GAP_THRESHOLD:
                is_close = True
                break
        
        if is_close:
            cv2.drawContours(component_mask, [component], -1, 255, -1)
            accepted_rects.append(rect)

    filtered_mask = cv2.bitwise_and(component_mask, binary_mask.copy())

    roi_slice, offset = get_roi_bounding_box(filtered_mask)
    
    if roi_slice is None:
        return np.zeros_like(binary_mask), 0
    
    # Crop -> Skeletonize -> Prune -> Reconstruct
    cropped_mask = filtered_mask[roi_slice]
    skeleton_crop = np.array(skeletonize(cropped_mask // 255), dtype='uint8')
    
    # Apply your cleanup pipeline
    skeleton_crop = trim(prune(skeleton_crop, 5))
    skeleton_crop = trim(prune(skeleton_crop, 3))
    
    # Place back into full frame
    full_skeleton = np.zeros_like(binary_mask)
    full_skeleton[roi_slice] = skeleton_crop
    
    return full_skeleton, np.sum(skeleton_crop)

def extract_skeleton(binary_mask):
    roi_slice, offset = get_roi_bounding_box(binary_mask)
    
    if roi_slice is None:
        # Empty mask
        return np.zeros_like(binary_mask), np.array([]), np.array([]), False

    # Perform operations on the smaller crop
    cropped_mask = binary_mask[roi_slice]
    
    # Convert binary mask to 1-pixel-wide skeleton
    skeleton_crop = np.array(skeletonize(cropped_mask // 255), dtype='uint8')

    # First cleanup pass
    skeleton_crop = prune(skeleton_crop, 5)  
    skeleton_crop = trim(skeleton_crop)
    
    # Second cleanup pass
    skeleton_crop = prune(skeleton_crop, 3)  
    skeleton_crop = trim(skeleton_crop)
    
    # Identify branch points and end points on the crop
    branch_points, end_points = skeleton_nodes(skeleton_crop)
    
    # Adjust coordinates back to full image space
    if len(branch_points) > 0:
        branch_points[:, 0] += offset[0] # x
        branch_points[:, 1] += offset[1] # y
        
    if len(end_points) > 0:
        end_points[:, 0] += offset[0] # x
        end_points[:, 1] += offset[1] # y

    # Reconstruct full size skeleton image
    full_skeleton = np.zeros_like(binary_mask)
    full_skeleton[roi_slice] = skeleton_crop
    
    # Valid skeleton must have at least 2 endpoints (seed + at least one tip)
    is_valid = len(end_points) >= 2
    
    return full_skeleton, branch_points, end_points, is_valid

def trim(ske): ## Removes unwanted pixels from the skeleton
    # Using pre-allocated global kernels
    bp = np.zeros_like(ske)
    for t in T_KERNELS:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    return ske


_RING = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def _ring_components(sk, y, x):
    """8-neighborhood foreground pixels grouped into connected branches."""
    h, w = sk.shape
    vals = []
    for dy, dx in _RING:
        yy, xx = y + dy, x + dx
        if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
            vals.append((int(yy), int(xx)))
        else:
            vals.append(None)
    if vals and all(v is not None for v in vals):
        return [vals]
    comps = []
    cur = []
    for v in vals:
        if v is None:
            if cur:
                comps.append(cur)
                cur = []
        else:
            cur.append(v)
    if cur:
        if comps and vals[0] is not None:
            comps[0] = cur + comps[0]
        else:
            comps.append(cur)
    return comps

def _branch_counts(sk):
    """How many separate branches leave each pixel (0 on background).

    1 = tip, 2 = along a branch, 3 or more = junction.
    Diagonal contacts that belong to the same branch count as one.
    """
    padded = np.pad(sk, 1, constant_values=0)
    neighbors = (
        padded[:-2, 1:-1],  # N
        padded[:-2, 2:],    # NE
        padded[1:-1, 2:],   # E
        padded[2:, 2:],     # SE
        padded[2:, 1:-1],   # S
        padded[2:, :-2],    # SW
        padded[1:-1, :-2],  # W
        padded[:-2, :-2],   # NW
    )
    counts = np.zeros(sk.shape, np.uint8)
    for i, nbr in enumerate(neighbors):
        nxt = neighbors[(i + 1) % 8]
        counts += (nbr == 0) & (nxt > 0)
    counts[sk == 0] = 0
    return counts

def _follow_branch(sk, counts, tip, max_len):
    """Pixels of the branch that starts at tip, excluding the junction.

    Returns None when the branch is longer than max_len or never meets a junction.
    """
    path = [tip]
    prev = None
    cur = tip
    while True:
        comps = _ring_components(sk, cur[0], cur[1])
        forward = [c for c in comps if prev is None or prev not in c]
        if len(forward) != 1:
            return None
        step = forward[0]
        if any(counts[pixel] >= 3 for pixel in step):
            return path
        if len(step) != 1 or len(path) >= max_len:
            return None
        prev, cur = cur, step[0]
        path.append(cur)

def prune(skel, num_it):
    """Remove spurs of at most num_it pixels.

    Each tip starts a branch. The branch is the pixels walked until the first
    junction. A short branch is cleared and the junction stays, so two short
    tips become one end. The main root and longer laterals are not touched.
    """
    sk = (np.asarray(skel) > 0).astype(np.uint8)
    if num_it <= 0 or not np.any(sk):
        return sk

    counts = _branch_counts(sk)
    tip_rows, tip_cols = np.where(counts == 1)
    spurs = []
    for y, x in zip(tip_rows.tolist(), tip_cols.tolist()):
        branch = _follow_branch(sk, counts, (y, x), int(num_it))
        if branch is not None:
            spurs.extend(branch)

    for y, x in spurs:
        sk[y, x] = 0
    return sk


def endPoints(skel):
    ep = np.zeros_like(skel)
    # Use global EP_KERNELS
    for kernel in EP_KERNELS:
        ep = cv2.add(ep, cv2.morphologyEx(skel, cv2.MORPH_HITMISS, kernel))
    return ep


def skeleton_nodes(ske):
    branch = branchedPoints(ske)
    end = endPoints(ske)
    
    bp = np.where(branch == 1)
    bnodes = []
    for i in range(len(bp[0])):
        bnodes.append([bp[1][i],bp[0][i]])
    
    ep = np.where(end == 1)
    enodes = []
    for i in range(len(ep[0])):
        enodes.append([ep[1][i],ep[0][i]])
    
    return np.array(bnodes), np.array(enodes)


def branchedPoints(skel):
    bp = np.zeros(skel.shape, dtype=int)
    
    for x in X_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T_BRANCH_KERNELS:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp