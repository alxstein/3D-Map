import cv2
import numpy as np

ALPHA = 0.04
KERNEL_SIZE = 5
THRESHOLD_RATIO = 0.01

def harris_response(img):
    
    # --------------------------------------------------------------------------------
    # DERIVATIVES
    # --------------------------------------------------------------------------------

    # 1. Gradients
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)

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

def main():

    # --------------------------------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------------------------------

    zimmer1_img = cv2.imread(r'..\Images\Zimmer1.jpg', cv2.IMREAD_GRAYSCALE)
    zimmer2_img = cv2.imread(r'..\Images\Zimmer2.jpg', cv2.IMREAD_GRAYSCALE)

    new_width = 1080
    new_height = 1440
    
    # Resize the image
    zimmer1_resized = cv2.resize(zimmer1_img, (new_width, new_height))
    zimmer2_resized = cv2.resize(zimmer2_img, (new_width, new_height))

    # Get corners
    z1_corners_x, z1_corners_y = find_corner_coordinates(zimmer1_resized)
    z2_corners_x, z2_corners_y = find_corner_coordinates(zimmer2_resized)

    # --------------------------------------------------------------------------------
    # Visualize
    # --------------------------------------------------------------------------------
    apply_corners(zimmer1_resized, z1_corners_x, z1_corners_y)
    apply_corners(zimmer2_resized, z2_corners_x, z2_corners_y)

    window = "Visualisierung"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, 480, 640)

    cv2.imshow(window, zimmer2_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()