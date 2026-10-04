import cv2
import numpy as np

ALPHA = 0.02
KERNEL_SIZE = 3
THRESHOLD_RATIO = 0.03
DOGS_SCALES = [1, 1.4, 2, 2.8, 4, 5.6]
PATCH_SIZE = 16
SUBPATCH_SIZE = 4
SIGMA_WEIGHT = 8
MATCHING_THRESHOLD = 0.75


# --------------------------------------------------------------------------------
# HARRIS DETECTOR
# --------------------------------------------------------------------------------

def harris_response(img):

    # 1. Gradients
    Ix, Iy = get_gradients(img)

    # 2. Products
    Ix2 = Ix * Ix
    IxIy = Ix * Iy
    Iy2 = Iy * Iy

    # 3. Gaussian Filtering
    Sxx = cv2.GaussianBlur(Ix2, (5, 5), 0)
    Syy = cv2.GaussianBlur(Iy2, (5, 5), 0)
    Sxy = cv2.GaussianBlur(IxIy, (5, 5), 0)

    return Sxx, Syy, Sxy

def cornerness_function(Sxx, Syy, Sxy):
    return Sxx * Syy - (Sxy ** 2) - ALPHA * ((Sxx + Syy) ** 2)

def non_maximum_suppression(corners, threshold_ratio=THRESHOLD_RATIO, window_size=KERNEL_SIZE):
    # Compute threshold
    threshold = threshold_ratio * corners.max()

    # Strong Responses
    strong = corners > threshold

    # Compute local max
    kernel = np.ones((window_size, window_size), np.uint8)
    local_max = cv2.dilate(corners, kernel)

    corners = (corners == local_max) & strong

    return corners

def find_corner_coordinates(img):
    Sxx, Syy, Sxy = harris_response(img)
    img = cornerness_function(Sxx, Syy, Sxy)
    img = non_maximum_suppression(img)

    y, x = np.where(img)

    return x, y

def apply_corners(img, corners_x, corners_y):
    for x_i, y_i in zip(corners_x, corners_y):
        cv2.circle(img, (x_i, y_i), 5, 255, -1)

def get_dogs(img):
    img = img.astype(np.float32) / 255.0

    # Get blurred images
    sigmas = DOGS_SCALES
    scale_space = []

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )
        scale_space.append(blurred)

    # Compute difference of gaussians
    dog_space = []

    for i in range(len(scale_space) - 1):
        dog = scale_space[i + 1] - scale_space[i]
        dog_space.append(dog)

    return scale_space, dog_space

def find_scale_space_extrema(dog_space, threshold=THRESHOLD_RATIO):
    keypoints = []

    for s in range(1, len(dog_space) - 1):

        dog_prev = dog_space[s - 1]
        dog_curr = dog_space[s]
        dog_next = dog_space[s + 1]

        height, width = dog_curr.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                value = dog_curr[y, x]

                # Get only strong responses
                if abs(value) < threshold:
                    continue

                # 3x3 neighborhood of previous scale
                neighbors_prev = dog_prev[y-1:y+2, x-1:x+2]

                # 3x3 neighborhood of current scale
                neighbors_curr = dog_curr[y-1:y+2, x-1:x+2]

                # 3x3 neighborhood of next scale
                neighbors_next = dog_next[y-1:y+2, x-1:x+2]

                # Combine neighbors
                neighbors = np.concatenate([
                    neighbors_prev.ravel(),
                    neighbors_curr.ravel(),
                    neighbors_next.ravel()
                ])

                # remove current center
                center_index = 9 + 4
                neighbors = np.delete(neighbors, center_index)

                # Check extremum
                if value > neighbors.max() or value < neighbors.min():
                    keypoints.append((x, y, s, value))

    return keypoints

def get_gradients(img):
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)

    return Ix, Iy

def assign_orientation(keypoints, scale_space, magnitudes, orientations):    
    sift_keypoints = []
    for (x, y, scale, dog_value) in keypoints:
        bins = np.zeros(36)
        dominant_orientation = []
        sigma = DOGS_SCALES[scale]
        radius = int(3 * sigma)

        # Compute histogram of gradient for current keypoint
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x2 = x + dx
                y2 = y + dy

                if x2 < 0 or x2 >= scale_space[scale].shape[1] or y2 < 0 or y2 >= scale_space[scale].shape[0]:
                    continue

                # Add value to bin
                bin = int(orientations[scale][y2, x2] / 10)
                weight = np.exp(
                    -(dx**2 + dy**2) / (2 * sigma**2)
                )
                bins[bin] += weight * magnitudes[scale][y2, x2]

        # Determine dominant orientation(s) - get all dominant orientations above 80% of max bin value
        threshold = 0.8 * bins.max()
        dominant_orientation = np.where(bins > threshold)[0] * 10

        # For multiple dominant orientations, create multiple keypoints
        for keypoint_orientation in dominant_orientation:
            keypoint = {
                "x": x,
                "y": y,
                "scale": scale,
                "orientation": keypoint_orientation,
                "descriptor": None
            }
            sift_keypoints.append(keypoint)

    return sift_keypoints

def compute_descriptors(sift_keypoints, magnitudes, orientations):
    valid_descriptors = []
    for keypoint in sift_keypoints:
        center_x = keypoint["x"]
        center_y = keypoint["y"]
        half_size = int(PATCH_SIZE / 2)
        height, width = magnitudes[0].shape
        
        if center_x - half_size < 0 or center_x + half_size > width or center_y - half_size < 0 or center_y + half_size > height:
            continue

        scale = keypoint["scale"]
        keypoint_orientation = keypoint["orientation"]

        descriptor = []

        for cell_y in range(4):
            for cell_x in range(4):

                histogram = np.zeros(8)

                for dy in range(SUBPATCH_SIZE):
                    for dx in range(SUBPATCH_SIZE):

                        yy = center_y - half_size + cell_y * SUBPATCH_SIZE + dy
                        xx = center_x - half_size + cell_x * SUBPATCH_SIZE + dx

                        magnitude = magnitudes[scale][yy, xx]
                        weight = np.exp(
                            -((center_x-xx)**2 + (center_y-yy)**2) / (2 * SIGMA_WEIGHT**2)
                        )
                        angle = orientations[scale][yy, xx]

                        # Normalize orientation
                        angle = (angle - keypoint_orientation) % 360
                        bin_index = int(angle / 45)

                        histogram[bin_index] += weight * magnitude

                descriptor.extend(histogram)

        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        keypoint["descriptor"] = descriptor
        valid_descriptors.append(keypoint)

    return valid_descriptors

def compute_corner_descriptors(img):
    # 1. Compute relevant keypoints
    scale_space, dogs = get_dogs(img)
    keypoints = find_scale_space_extrema(dogs)

    # 2. Compute orientation
    # Compute magnitudes and gradients for each scale space
    magnitudes = []
    orientations = []
    for blurred_img in scale_space:
        Ix, Iy = get_gradients(blurred_img)
        magnitude, orientation = cv2.cartToPolar(Ix, Iy, angleInDegrees=True)
        magnitudes.append(magnitude)
        orientations.append(orientation)
    sift_keypoints = assign_orientation(keypoints, scale_space, magnitudes, orientations)

    # 3. Compute feature descriptors
    sift_keypoints = compute_descriptors(sift_keypoints, magnitudes, orientations)

    return sift_keypoints

def match_descriptors(img1_keypoints, img2_keypoints):

    descriptors_A = np.array([
        keypoint["descriptor"]
        for keypoint in img1_keypoints
    ], dtype=np.float32)

    descriptors_B = np.array([
        keypoint["descriptor"]
        for keypoint in img2_keypoints
    ], dtype=np.float32)

    # Squared Euclidean distances
    A_squared = np.sum(descriptors_A ** 2, axis=1, keepdims=True)
    B_squared = np.sum(descriptors_B ** 2, axis=1, keepdims=True).T

    distances_squared = (
        A_squared
        + B_squared
        - 2 * descriptors_A @ descriptors_B.T
    )

    # Numerische Rundungsfehler verhindern
    distances_squared = np.maximum(distances_squared, 0)

    # Zwei kleinste Distanzen finden
    nearest_indices = np.argpartition(
        distances_squared,
        kth=1,
        axis=1
    )[:, :2]

    # Die beiden gefundenen Kandidaten nach Distanz sortieren
    row_indices = np.arange(len(descriptors_A))

    first = nearest_indices[:, 0]
    second = nearest_indices[:, 1]

    swap = (
        distances_squared[row_indices, second]
        < distances_squared[row_indices, first]
    )

    best_indices = first.copy()
    second_indices = second.copy()

    best_indices[swap] = second[swap]
    second_indices[swap] = first[swap]

    best_distances = np.sqrt(
        distances_squared[row_indices, best_indices]
    )

    second_distances = np.sqrt(
        distances_squared[row_indices, second_indices]
    )

    # Lowe Ratio Test
    ratios = best_distances / (second_distances + 1e-12)
    good_matches = ratios < MATCHING_THRESHOLD

    matches = []

    for idx_A in range(len(img1_keypoints)):

        if not good_matches[idx_A]:
            continue

        idx_B = best_indices[idx_A]

        matches.append({
            "idx_A": idx_A,
            "idx_B": idx_B,
            "distance": best_distances[idx_A],
            "ratio": ratios[idx_A]
        })

    return matches

def visualize_matches(img_A, img_B, keypoints_A, keypoints_B, matches):

    # Convert pictures to BGR to make colored points
    if len(img_A.shape) == 2:
        img_A = cv2.cvtColor(img_A, cv2.COLOR_GRAY2BGR)

    if len(img_B.shape) == 2:
        img_B = cv2.cvtColor(img_B, cv2.COLOR_GRAY2BGR)

    height_A, width_A = img_A.shape[:2]
    height_B, width_B = img_B.shape[:2]

    height = max(height_A, height_B)

    canvas = np.zeros(
        (height, width_A + width_B, 3),
        dtype=np.uint8
    )

    canvas[:height_A, :width_A] = img_A
    canvas[:height_B, width_A:width_A + width_B] = img_B

    # Visualize all corners
    for keypoint in keypoints_A:
        x = int(keypoint["x"])
        y = int(keypoint["y"])

        cv2.circle(
            canvas,
            (x, y),
            4,
            (0, 255, 0),
            1
        )

    for keypoint in keypoints_B:
        x = int(keypoint["x"]) + width_A
        y = int(keypoint["y"])

        cv2.circle(
            canvas,
            (x, y),
            4,
            (0, 255, 0),
            1
        )

    # Connect matched points
    for match in matches:

        idx_A = match["idx_A"]
        idx_B = match["idx_B"]

        kp_A = keypoints_A[idx_A]
        kp_B = keypoints_B[idx_B]

        x_A = int(kp_A["x"])
        y_A = int(kp_A["y"])

        x_B = int(kp_B["x"]) + width_A
        y_B = int(kp_B["y"])

        cv2.line(
            canvas,
            (x_A, y_A),
            (x_B, y_B),
            (0, 0, 255),
            1
        )

        # Highlight matched points
        cv2.circle(
            canvas,
            (x_A, y_A),
            5,
            (255, 0, 0),
            2
        )

        cv2.circle(
            canvas,
            (x_B, y_B),
            5,
            (255, 0, 0),
            2
        )

    return canvas

def main():

    # --------------------------------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------------------------------

    zimmer1_img = cv2.imread(r'..\Images\Schrank1.jpg', cv2.IMREAD_GRAYSCALE)
    zimmer2_img = cv2.imread(r'..\Images\Schrank2.jpg', cv2.IMREAD_GRAYSCALE)

    new_width = 1080
    new_height = 1440
    
    # Resize the image
    zimmer1_resized = cv2.resize(zimmer1_img, (new_width, new_height))
    zimmer2_resized = cv2.resize(zimmer2_img, (new_width, new_height))
    z1_copy = zimmer1_resized.copy()
    z2_copy = zimmer2_resized.copy()

    # --------------------------------------------------------------------------------
    # SIFT
    # --------------------------------------------------------------------------------

    # 1. Compute descriptors
    z1_keypoints = compute_corner_descriptors(z1_copy)
    z2_keypoints = compute_corner_descriptors(z2_copy)
    
    # 2. Match descriptors
    matches = match_descriptors(z1_keypoints, z2_keypoints)
    # --------------------------------------------------------------------------------
    # Visualize
    # --------------------------------------------------------------------------------

    # window = "Visualisierung"
    # cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    # cv2.resizeWindow(window, 480, 640)

    # cv2.imshow(window, zimmer1_resized)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    visualization = visualize_matches(
        zimmer1_resized,
        zimmer2_resized,
        z1_keypoints,
        z2_keypoints,
        matches
    )
    window = "Feature Matches"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, 480 * 2, 640)
    cv2.imshow(window, visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()