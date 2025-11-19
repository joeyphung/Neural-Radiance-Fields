***

## Usage
This project implements a complete pipeline for synthesizing novel views of a scene, starting from camera calibration using ArUco tags and ending with a fully trained NeRF model.

### Running the Pipeline
The project is split into data preparation (Python script) and model training (Jupyter Notebook).

* **Data Preparation (`camera_calibration.py`)**:
    This script must be run first to generate the training data. It processes raw images containing ArUco markers to determine camera intrinsics and poses.
    * **Visualization**: The script utilizes `viser` to visualize camera frustums in 3D space to verify pose estimation accuracy before training.
    * **Output**: It generates a `.npz` file containing training/validation splits of images, camera-to-world matrices, and focal lengths.

* **Model Training (`nerf.ipynb`)**:
    The notebook loads the `.npz` data and trains the implicit neural representation. It includes cells for:
    1.  **Ray Generation**: Converting pixels to 3D rays using linear algebra helpers.
    2.  **Training Loop**: Optimizing the MLP using Mean Squared Error.
    3.  **Rendering**: Generating videos of the object from novel camera trajectories.

### `camera_calibration.py`
This script handles the transition from physical images to mathematically aligned data required for NeRF.

* `calibrate(dir_path, tag_corners, ...)`: Computes the intrinsic camera parameters.
    * **Input**: A directory of calibration images and the expected 3D coordinates of ArUco tags.
    * **Process**: It detects ArUco markers in grayscale images. Using the correspondence between the detected 2D image points and known 3D tag corners, it solves for the camera matrix and distortion coefficients using `cv.calibrateCamera`.
    * **Output**: Returns the intrinsic `cameraMatrix` and `distCoeffs`.

* `estimate_pose(dir_path, cameraMatrix, ...)`: Calculates the extrinsic parameters (Camera-to-World) for each image.
    * **Input**: Image directory and the calibrated intrinsic matrix.
    * **Process**: It detects markers in the scene images. Using `cv.solvePnP`, it computes the rotation ($R$) and translation ($t$) vectors. It constructs the 4x4 transformation matrix:
        $$C2W = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$$
    * **Output**: Returns a list of valid BGR images and their corresponding $4 \times 4$ C2W matrices.

* `save_imgs(undistorted_imgs, cameraMatrix, c2ws, ...)`: Formats and saves the dataset.
    * **Input**: List of processed images, camera matrices, and split ratios.
    * **Process**: It resizes images to a target dimension, converts them to RGB, and splits them into training, validation, and test sets. It also calculates a unified focal length $f$ by averaging $f_x$ and $f_y$.
    * **Output**: Saves a compressed `my_data.npz` file containing arrays for images, poses, and the focal length.

### `nerf.ipynb`
This notebook contains the deep learning implementation, defining the ray marching logic and the Multi-Layer Perceptron (MLP).

#### Ray Generation Helpers
These functions handle the linear algebra required to convert 2D image pixels into 3D rays in world space.

* `pixel_to_camera(K, uv, s)`: Unprojects pixels into camera coordinate space.
    * **Input**: Intrinsics matrix $K$, pixel coordinates $uv$, and a depth scale $s$.
    * **Process**: It constructs homogeneous coordinates $[u, v, 1]$ and multiplies by the inverse of the intrinsic matrix ($x_c = s \cdot K^{-1} \cdot uv_{hom}$).
    * **Output**: 3D points relative to the camera origin.

* `transform(c2w, x_c)`: Converts points from camera space to world space.
    * **Input**: Camera-to-World matrix `c2w` and camera-space points `x_c`.
    * **Process**: Applies the rigid body transformation using the rotation sub-matrix $R$ and translation vector $t$:
      $$x_w = R \cdot x_c + t$$
    * **Output**: 3D points in the world coordinate system.

* `pixel_to_ray(K, c2w, uv)`: Orchestrates the full pixel-to-ray conversion.
    * **Input**: Intrinsics $K$, camera pose $c2w$, and pixel coords $uv$.
    * **Process**:
        1.  Identifies the camera origin ($r_o$) using the translation vector of the pose.
        2.  Calls `pixel_to_camera` and `transform` to find the pixel's position in world space ($x_w$).
        3.  Calculates the normalized direction vector: $r_d = \frac{x_w - r_o}{||x_w - r_o||}$.
    * **Output**: Returns ray origins $r_o$ and ray directions $r_d$.

#### Core Classes & Training
* `RaysData.sample_rays(N)`: Generates training minibatches.
    * **Input**: The desired batch size $N$ (number of rays).
    * **Process**: Uses `torch.randperm` to generate random indices and selects a random subset of rays (origins, directions) and their corresponding ground truth pixel colors from the pre-calculated dataset. This enables Stochastic Gradient Descent (SGD).
    * **Output**: A batch of $N$ rays and colors for training.

* `sample_along_rays(rays_o, rays_d, ...)`: Performs stratified sampling.
    * **Input**: Ray origins/directions, near/far bounds, and number of samples.
    * **Process**: It generates depth values $t$ between the near and far planes. During training, it adds random noise (perturbation) to these values to allow the network to learn a continuous representation.
    * **Output**: A tensor of 3D query points ($x = o + t \cdot d$).

* `NERF_3D(x, ray_d)`: The neural network architecture.
    * **Input**: A batch of 3D spatial coordinates $x$ and viewing directions $d$.
    * **Process**:
        1.  Applies Positional Encoding (PE) to map inputs to higher frequency domains ($\sin, \cos$).
        2.  Passes encoded $x$ through dense layers with ReLU activations.
        3.  Outputs volume density $\sigma$.
        4.  Concatenates geometric features with encoded direction $d$ to predict RGB color.
    * **Output**: Predicted color $c$ and density $\sigma$ for every sample point.

* `volrend(sigmas, rgbs, step_size)`: Implements the volume rendering integral.
    * **Input**: Densities $\sigma$ and colors $c$ predicted by the network.
    * **Process**: Calculates transmittance $T_i = \exp(-\sum \sigma_j \delta_j)$ to determine how much light is blocked by earlier samples, then aggregates color along the ray.
    * **Output**: The final rendered pixel color.

* `train_loop(...)`: Orchestrates the optimization.
    * **Input**: Model, optimizer, and data loaders.
    * **Process**: Iteratively calls `sample_rays` to get a batch, renders them using the model, computes Mean Squared Error (MSE) loss, and updates weights.
    * **Output**: Plots of PSNR and Loss curves over epochs.

### Helper Files
* **`utils.py`**: A utility module handling low-level image operations.
    * `load_img`/`save_img`: Handles reading/writing with OpenCV.
    * `display_img`: Uses `matplotlib` to show images within notebooks.
    * `float64_to_uint8`: Converts normalized float image arrays back to standard byte formats.