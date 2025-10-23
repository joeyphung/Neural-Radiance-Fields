## Usage
This project is divided into several files, each responsible for a specific part of the image processing pipeline.

### Running the Scripts
Each executable script has a global `command_line` flag that can be toggled to change how arguments are provided.

* **`command_line = False`**: The script will use the file paths and settings that are manually set within the file. This is useful for development and testing.
* **`command_line = True`**: The script must be run from the terminal and will expect arguments (like image paths) to be provided directly on the command line.

During runtime, the scripts will **display important intermediate and final images** in pop-up windows for immediate visualization. All generated output images are also **saved to the `./out/` directory**, organized into subfolders corresponding to the task.

### `homography.py`
This script computes the homography matrix, which is a 3x3 transformation matrix that maps points from one image plane to another.

* `computeH(img1_pts, img2_pts)`: This is the core function for calculating the homography.
    * **Input**: It takes two lists of corresponding points (`img1_pts` and `img2_pts`), where each point is an (x, y) tuple. At least 4 point correspondences are needed.
    * **Process**: It sets up a system of linear equations ($Ah = b$) based on the point correspondences and solves for the 8 unknown parameters of the homography matrix using the least squares method (`np.linalg.lstsq`).
    * **Output**: It returns a 3x3 NumPy array representing the homography matrix $H$.

### `warp.py`
This file contains functions to apply a homography transformation to an image, effectively "warping" it into a new perspective. It provides two different interpolation methods.

* `warpImageNearestNeighbor(img, H, out_dim=None)`: Warps an image using nearest-neighbor interpolation. For each pixel in the output image, it calculates the corresponding location in the source image and simply takes the value of the closest pixel. This method is fast but can result in blocky or jagged artifacts.
* `warpImageBilinear(img, H, out_dim=None)`: Warps an image using bilinear interpolation. For each pixel in the output image, it finds the corresponding non-integer location in the source image and calculates a weighted average of the four nearest pixel values. This produces a smoother, higher-quality result than nearest-neighbor interpolation.

### `rectify.py`
This script handles image rectification, which adjusts an image to correct perspective distortion. It makes tilted or angled surfaces appear flat and front-facing.

* `rectify(img, img_name, img_pts)`: This function orchestrates the rectification process.
    * **Input**: It takes the source image (`img`), its name, and a list of 4 points (`img_pts`) that define the corners of the planar region to be rectified.
    * **Process**: It defines a corresponding set of points that form a rectangle. It then calls `computeH` to find the homography that maps the user-selected points to the corners of the destination rectangle. Finally, it uses one of the warp functions from `warp.py` to transform the image.
    * **Output**: The rectified image.

### `mosaic.py`
This script stitches two images together to create a single panoramic mosaic. It aligns the images using a computed homography and blends them seamlessly.

* `mosaic(img1, img2, img1_name, img2_name, img1_pts, img2_pts, ...)`: This function creates the image mosaic.
    * **Input**: It requires two images (`img1`, `img2`) and their corresponding points (`img1_pts`, `img2_pts`) for alignment.
    * **Process**:
        1.  Calculates the homography matrix $H$ using `computeH` to map `img1` onto the plane of `img2`.
        2.  Warps `img1` using the calculated homography.
        3.  Shifts `img2` to align with the warped `img1` in a larger canvas.
        4.  Pads both images to ensure they are the same size.
        5.  Uses multi-resolution blending (from `multiresolution_blending.py`) to smoothly combine the overlapping regions of the two images, avoiding visible seams.
    * **Output**: The final blended mosaic image.

### `harris.py`
This script is responsible for the initial interest point detection in an image.

* `get_harris_corners(im)`: This function identifies corners, which are regions in the image with strong intensity variation in all directions.
    * **Input**: A grayscale image.
    * **Process**: It applies the Harris corner detection algorithm to find interest points. It then refines these locations to find local maxima and discards any points found too close to the image border.
    * **Output**: Returns an array of `(y, x)` coordinates for the detected Harris corners.

### `corner_detection.py`
This script filters the corners detected by `harris.py` to get the best points while remaining spatially distributed.

* `ANMS(img, num_pts)`: This function implements **Adaptive Non-Maximal Suppression**.
    * **Input**: An image and the desired number of interest points (`num_pts`).
    * **Process**: It takes the vast number of corners from the Harris detector and selects the best `num_pts` of them. The selection criteria ensures that the chosen points are not just the strongest corners but are also spread out across the entire image, which is crucial for calculating an accurate homography.
    * **Output**: An array containing the `(y, x)` coordinates of the `num_pts` best-distributed corners.

### `feature_extraction.py`
This script creates a robust descriptor for each interest point.

* `get_features(img, num_pts)`: This function generates a feature vector for each keypoint from ANMS.
    * **Input**: An image and the number of points to pass into ANMS.
    * **Process**: For each ANMS point, it extracts a 40x40 pixel patch from the image. This patch is then downsampled to 8x8. To make the feature invariant to lighting changes, it is normalized by subtracting the mean and dividing by the standard deviation. Finally, the 8x8 patch is flattened into a 64-element feature vector.
    * **Output**: An array of feature vectors and an array of the corresponding ANMS points.

### `feature_matching.py`
This script finds corresponding features between two images.

* `get_matching_features(img1, img2, num_pts, threshold)`: This is the core function for finding point correspondences.
    * **Input**: Two images, the number of points, and a threshold for the ratio test.
    * **Process**: It extracts feature vectors from both images. For each feature in the first image, it finds its two nearest neighbors in the second image using a k-d tree for efficiency. It then applies **Lowe's ratio test** by comparing the distance to the nearest neighbor with the distance to the second-nearest neighbor. If the ratio is below the `threshold`, the match is considered strong and unambiguous.
    * **Output**: Two arrays of `(y, x)` coordinates that represent the matched points between the two images.

### `RANSAC.py`
This script computes a robust homography by filtering out incorrect matches (outliers).

* `RANSAC(img1, img2, ..., iterations, epsilon)`: This function implements the **RANdom SAmple Consensus** algorithm.
    * **Input**: The matched points from the previous step, the number of iterations to run, and an `epsilon` threshold for classifying inliers.
    * **Process**: It iteratively computes a homography matrix from a random sample of 4 point pairs. It then tests this homography against all other matched pairs. Any pair where the transformed point from the first image lands within `epsilon` pixels of its corresponding point in the second image is counted as an "inlier." After many iterations, the algorithm returns the set of inliers from the model that had the most support.
    * **Output**: The cleaned, robust set of corresponding points (inliers) that can be used to compute the final, accurate homography for stitching.

### Helper Files
* **`point_picker.py`**: Provides functions for users to interactively select corresponding points on images using `matplotlib`.
* **`multiresolution_blending.py`**: Contains the logic for blending two images smoothly using Laplacian and Gaussian pyramids. This technique ensures that the seam between the stitched images is not visible.
* **`utils.py`**: A utility module containing helper functions for common tasks such as loading, displaying, and saving images, as well as type conversions between `uint8` and `float64`.