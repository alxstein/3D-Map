import cv2
import numpy as np
from sift import compute_corner_descriptors, match_descriptors

RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 4.0

def ransac(images):
    # Assumes images are already preprocessed
    # images: list of cv2 images

    # --------------------------------------------------------------------------------
    # RANSAC PIPELINE
    # --------------------------------------------------------------------------------

    # Check if list is too small
    if len(images) < 2:
        print("Amount of images passed to RANSAC are less than 2.")
        return []

    # Resulting matching points between two frames - one element at index i is the matched points between images[i] and images[i+1]
    frame_matched_keypoints = []

    keypoints_A = compute_corner_descriptors(images[0])
    for i in range(len(images) - 1):
        # First get the keypoints and matches from the current and next frame
        keypoints_B = compute_corner_descriptors(images[i+1])

        matches = match_descriptors(keypoints_A, keypoints_B)

        # Apply ransac now
        best_F, best_inliers = ransac_matching(keypoints_A, keypoints_B, matches)
        
        if best_inliers != []:
            frame_matched_keypoints.append(best_inliers)

        # keypoints_B is now the first frame, so we switch to A
        keypoints_A = keypoints_B

    return frame_matched_keypoints

def ransac_matching(keypoints_A, keypoints_B, matches):

    # Get coordinates of the keypoints that match in both images: points A and B have same length, points_A[i] is the point that was matched to points_B[i]
    points_A, points_B = extract_coordinates(keypoints_A, keypoints_B, matches)

    best_F = None
    best_inlier_indices = np.array([], dtype=int)
    best_inlier_count = 0

    for _ in range(RANSAC_ITERATIONS):
        # Randomly select 8 matches
        samples_A, samples_B = random_8_matches(points_A, points_B)

        # Now compute the model to describe the scenery, the fundamental matrix
        F = estimate_fundamental_matrix(samples_A, samples_B)

        if F is None:
            continue

        # Measure how good the model fits all matched points
        errors = compute_sampson_errors(F, points_A, points_B)

        # Classify matches as inliers using the threshold
        inlier_mask = errors < RANSAC_THRESHOLD
        inlier_indices = np.flatnonzero(inlier_mask)
        inlier_count = len(inlier_indices)

        if inlier_count > best_inlier_count:
            best_F = F
            best_inlier_indices = inlier_indices
            best_inlier_count = inlier_count

    if best_F is None or best_inlier_count < 8:
        return None, []

    # Return the exact keypoints from the best indices
    best_inliers = [matches[index] for index in best_inlier_indices]
    
    return best_F, best_inliers

def extract_coordinates(keypoints_A, keypoints_B, matches):
    # Extracts the coordinates of keypoint matches
    points_A = []
    points_B = []

    for match in matches:
        keypoint_A = keypoints_A[match["idx_A"]]
        coordinates_A = [keypoint_A["x"], keypoint_A["y"]]

        keypoint_B = keypoints_B[match["idx_B"]]
        coordinates_B = [keypoint_B["x"], keypoint_B["y"]]

        points_A.append(coordinates_A)
        points_B.append(coordinates_B)

    # Convert into np array    
    points_A = np.array(points_A)
    points_B = np.array(points_B)

    return points_A, points_B

def random_8_matches(points_A, points_B):
    # Get 8 random indices to get 8 matching points
    indices = np.random.choice(len(points_A), size=8, replace=False)

    samples_A = points_A[indices]
    samples_B = points_B[indices]

    return samples_A, samples_B

def estimate_fundamental_matrix(samples_A, samples_B):
    # Convert inputs to NumPy arrays
    # points_A = np.asarray(points_A, dtype=np.float64)
    # points_B = np.asarray(points_B, dtype=np.float64)

    # Validate the input shapes
    if samples_A.ndim != 2 or samples_A.shape[1] != 2:
        raise ValueError("points_A must have shape (N, 2).")

    if samples_B.ndim != 2 or samples_B.shape[1] != 2:
        raise ValueError("points_B must have shape (N, 2).")

    if samples_A.shape != samples_B.shape:
        raise ValueError("Both point arrays must have the same shape.")

    if len(samples_A) < 8:
        raise ValueError("At least 8 correspondences are required.")

    if not np.isfinite(samples_A).all() or not np.isfinite(samples_B).all():
        raise ValueError("Point coordinates must be finite.")

    # Step 1: Normalize points in both images
    normalized_A, T_A = normalize_points(samples_A)
    normalized_B, T_B = normalize_points(samples_B)

    x1 = normalized_A[:, 0]
    y1 = normalized_A[:, 1]
    x2 = normalized_B[:, 0]
    y2 = normalized_B[:, 1]

    # Step 2: Build the linear system A @ f = 0
    # Each row represents one point correspondence
    A = np.column_stack([
        x2 * x1,
        x2 * y1,
        x2,
        y2 * x1,
        y2 * y1,
        y2,
        x1,
        y1,
        np.ones(len(samples_A))
    ])

    # Step 3: Solve the system using SVD
    # The right singular vector for the smallest singular value
    # gives the least-squares solution
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    # Reshape the solution vector into a 3x3 matrix
    F_normalized = Vt[-1].reshape(3, 3)

    # Step 4: Enforce the rank-2 constraint on F
    U, singular_values, Vt_F = np.linalg.svd(F_normalized)

    # A valid fundamental matrix must have one zero singular value
    singular_values[-1] = 0.0

    F_normalized = U @ np.diag(singular_values) @ Vt_F

    # Step 5: Transform F back to the original image coordinates
    F = T_B.T @ F_normalized @ T_A

    # Step 6: Normalize the matrix scale for consistency
    matrix_norm = np.linalg.norm(F)

    if matrix_norm < 1e-12:
        raise ValueError("Failed to estimate a valid fundamental matrix.")

    F = F / matrix_norm

    return F

def normalize_points(points):
    """
    Normalize 2D points so that:
    - Their centroid is at the origin.
    - Their mean distance from the origin is sqrt(2).
    """
    # Calculate the centroid of the points.
    centroid = np.mean(points, axis=0)

    # Move the centroid to the origin.
    centered_points = points - centroid

    # Calculate the mean distance from the origin.
    mean_distance = np.mean(
        np.linalg.norm(centered_points, axis=1)
    )

    if mean_distance < 1e-12:
        raise ValueError("Cannot normalize points with zero spread.")

    # Scale the points so their mean distance becomes sqrt(2).
    scale = np.sqrt(2.0) / mean_distance

    # Construct the normalization transformation matrix.
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ])

    # Convert the points to homogeneous coordinates.
    points_h = np.column_stack([
        points,
        np.ones(len(points))
    ])

    # Apply the transformation and convert back to 2D.
    normalized_h = (T @ points_h.T).T
    normalized_points = normalized_h[:, :2]

    return normalized_points, T

def compute_sampson_errors(F, points_A, points_B):
    # Convert 2D points to homogeneous coordinates.
    points_A_h = np.column_stack([
        points_A,
        np.ones(len(points_A))
    ])
    points_B_h = np.column_stack([
        points_B,
        np.ones(len(points_B))
    ])

    # Compute the epipolar lines in both images.
    lines_B = (F @ points_A_h.T).T
    lines_A = (F.T @ points_B_h.T).T

    # Compute the squared algebraic error for each correspondence.
    residuals = np.sum(points_B_h * lines_B, axis=1)
    numerator = residuals ** 2

    # Compute the denominator of the Sampson error.
    denominator = (
        lines_B[:, 0] ** 2
        + lines_B[:, 1] ** 2
        + lines_A[:, 0] ** 2
        + lines_A[:, 1] ** 2
    )

    # Avoid division by zero for degenerate configurations.
    errors = np.full(len(points_A), np.inf)
    valid = denominator > 1e-12

    errors[valid] = numerator[valid] / denominator[valid]

    return errors

def main():

    zimmer1_img = cv2.imread(r'..\Images\Stuhl1.jpg', cv2.IMREAD_GRAYSCALE)
    zimmer2_img = cv2.imread(r'..\Images\Stuhl2.jpg', cv2.IMREAD_GRAYSCALE)

    new_width = 540
    new_height = 720
    
    # Resize the image
    zimmer1_resized = cv2.resize(zimmer1_img, (new_width, new_height))
    zimmer2_resized = cv2.resize(zimmer2_img, (new_width, new_height))
    z1_copy = zimmer1_resized.copy()
    z2_copy = zimmer2_resized.copy()

    images = [z1_copy, z2_copy]

    ransac(images)

if __name__ == "__main__":
    main()