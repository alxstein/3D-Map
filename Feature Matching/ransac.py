import cv2
import numpy as np
from sift import compute_corner_descriptors, match_descriptors

VIDEO_PATH = r"..\Videos\zimmer_video.mp4"
OUTPUT_PATH = r"..\Videos\ransac_output.mp4"

NEW_WIDTH = 540
NEW_HEIGHT = 720
FRAME_RATE = 20
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 4.0

def process_video(video_path, output_path):
    # Open the input video.
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Read the original video properties.
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if original_fps <= 0:
        cap.release()
        raise ValueError("Could not determine the original video FPS.")

    if FRAME_RATE <= 0 or FRAME_RATE > original_fps:
        cap.release()
        raise ValueError(
            f"FRAME_RATE must be between 1 and the original FPS ({original_fps:.2f})."
        )

    print(f"Original FPS: {original_fps:.2f}")
    print(f"Processing FPS: {FRAME_RATE}")
    print(f"Original frame count: {total_frames}")

    # Create the output video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        FRAME_RATE,
        (NEW_WIDTH * 2, NEW_HEIGHT)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    # Store the previous processed frame and its keypoints.
    previous_frame = None
    previous_keypoints = None

    # Store the track color for each keypoint in the previous frame.
    previous_track_colors = {}

    # Generate a new random color for a new track.
    def generate_track_color():
        return tuple(
            int(value)
            for value in np.random.randint(80, 256, size=3)
        )

    # Determine which source frames correspond to the target frame rate.
    frame_interval = original_fps / FRAME_RATE
    next_target_frame = 0
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Select frames at the requested processing rate.
            if frame_index + 1e-6 < next_target_frame:
                frame_index += 1
                continue

            next_target_frame += frame_interval
            frame_index += 1

            # Resize the selected frame to the required dimensions.
            frame = cv2.resize(
                frame,
                (NEW_WIDTH, NEW_HEIGHT)
            )

            # Convert the frame to grayscale for feature extraction.
            gray_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Extract keypoints for the current frame.
            current_keypoints = compute_corner_descriptors(gray_frame)

            # The first frame has no previous frame to match against.
            if previous_frame is None:
                previous_frame = gray_frame
                previous_keypoints = current_keypoints
                continue

            # Match descriptors between consecutive processed frames.
            matches = match_descriptors(
                previous_keypoints,
                current_keypoints
            )

            # Estimate the fundamental matrix and retrieve inliers.
            F, inliers = ransac_matching(
                previous_keypoints,
                current_keypoints,
                matches
            )

            print(
                f"Frame {frame_index}: "
                f"matches={len(matches)}, inliers={len(inliers)}"
            )

            # Draw only the inlier correspondences.
            # Convert both grayscale frames to BGR for colored visualization.
            frame_A = cv2.cvtColor(previous_frame, cv2.COLOR_GRAY2BGR)
            frame_B = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Create a canvas containing both frames side by side.
            visualization = cv2.hconcat([frame_A, frame_B])

            # Store the colors of the tracks that continue into the current frame.
            current_track_colors = {}

            # Draw only the inlier correspondences.
            for match in inliers:

                idx_A = match["idx_A"]
                idx_B = match["idx_B"]

                # Reuse the previous color if this keypoint belongs to an existing track.
                if idx_A in previous_track_colors:
                    color = previous_track_colors[idx_A]
                else:
                    # Assign a new color if this is a newly detected track.
                    color = generate_track_color()

                # Transfer the track color to the corresponding keypoint in the current frame.
                current_track_colors[idx_B] = color

                # Get the coordinates of both matched keypoints.
                x_A = int(previous_keypoints[idx_A]["x"])
                y_A = int(previous_keypoints[idx_A]["y"])

                x_B = int(current_keypoints[idx_B]["x"])
                y_B = int(current_keypoints[idx_B]["y"])

                # Shift the second point because the second frame is on the right.
                x_B_shifted = x_B + NEW_WIDTH

                # Draw the connecting line.
                cv2.line(
                    visualization,
                    (x_A, y_A),
                    (x_B_shifted, y_B),
                    color,
                    1
                )

                # Draw the matched points.
                cv2.circle(
                    visualization,
                    (x_A, y_A),
                    4,
                    color,
                    -1
                )

                cv2.circle(
                    visualization,
                    (x_B_shifted, y_B),
                    4,
                    color,
                    -1
                )

            # Only tracks matched in this frame are carried forward.
            previous_track_colors = current_track_colors

            # Write the visualization to the output video.
            writer.write(visualization)

            # Display the current visualization in a window.
            cv2.imshow("RANSAC Video", visualization)

            # Press Q to stop processing early.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Use the current frame as the next previous frame.
            previous_frame = gray_frame
            previous_keypoints = current_keypoints

    finally:
        # Release resources even if an error occurs.
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Visualization saved to: {output_path}")

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

        print("Number of matches:", len(matches))

        # Apply ransac now
        best_F, best_inliers = ransac_matching(keypoints_A, keypoints_B, matches)

        print("Fundamental matrix found:", best_F is not None)
        print("Number of inliers:", len(best_inliers))
        
        if best_inliers != []:
            frame_matched_keypoints.append(best_inliers)

        # keypoints_B is now the first frame, so we switch to A
        keypoints_A = keypoints_B

    return frame_matched_keypoints

def ransac_matching(keypoints_A, keypoints_B, matches):

    # Stop here if less than 8 matches have been found
    if len(matches) < 8:
        return None, []

    # Get coordinates of the keypoints that match in both images: points A and B have same length, points_A[i] is the point that was matched to points_B[i]
    points_A, points_B = extract_coordinates(keypoints_A, keypoints_B, matches)

    best_F = None
    best_inlier_indices = np.array([], dtype=int)
    best_inlier_count = 0

    for _ in range(RANSAC_ITERATIONS):
        # Randomly select 8 matches
        samples_A, samples_B = random_8_matches(points_A, points_B)

        # Now compute the model to describe the scenery, the fundamental matrix
        try:
            F = estimate_fundamental_matrix(samples_A, samples_B)

        except (ValueError, np.linalg.LinAlgError):
            # If 8 points have been selected that are extremely close to each other, the algorithm breaks and throws an error
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

    process_video(
        VIDEO_PATH,
        OUTPUT_PATH
    )

    # zimmer1_img = cv2.imread(r'..\Images\Stuhl1.jpg', cv2.IMREAD_GRAYSCALE)
    # zimmer2_img = cv2.imread(r'..\Images\Stuhl2.jpg', cv2.IMREAD_GRAYSCALE)

    # new_width = 540
    # new_height = 720
    
    # # Resize the image
    # zimmer1_resized = cv2.resize(zimmer1_img, (new_width, new_height))
    # zimmer2_resized = cv2.resize(zimmer2_img, (new_width, new_height))
    # z1_copy = zimmer1_resized.copy()
    # z2_copy = zimmer2_resized.copy()

    # images = [z1_copy, z2_copy]

    # ransac(images)

if __name__ == "__main__":
    main()