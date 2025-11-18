import cv2 as cv
import numpy as np
import viser
import time
import os
from utils import *
import math

# Returns the positions of the tags given a shape of (row_cnt, col_cnt) with each tag having a shape of (tag_size, tag_size). h_size is the horizontal gaps in the tags and v_size is the vertical gaps in the tags, all values are measured in meters
def get_tag_corners(tag_size, row_cnt, col_cnt, h_size=0, v_size=0):
    base_corner = np.array([
        [0, 0, 0],
        [tag_size, 0, 0],
        [tag_size, tag_size, 0],
        [0, tag_size, 0]
    ], dtype=np.float32)

    tag_corners = []
    for r in range(row_cnt):
        for c in range(col_cnt):
            offset = np.array([
                c * (h_size),
                r * (v_size),
                0
            ], dtype=np.float32)
            corner = base_corner + offset
            tag_corners.append(corner)
    return tag_corners

def calibrate(dir_path, tag_corners, expected_tags_calibrate):
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_params = cv.aruco.DetectorParameters()

    # Create the ArUco detector
    aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Each element of these lists represents one image
    objectPoints = []  # 3D points in world coordinates
    imagePoints = []  # 2D points in image plane

    # Loop through all your calibration images
    assert os.path.isdir(dir_path)
    imageSize = None
    for file_name in os.listdir(dir_path):
        
        file_path = dir_path + "/" + file_name
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE) # Convert to grayscale to enhance the detection process
        assert img is not None

        # Update imageSize to the size of the first image and make sure theyre all the same shape
        if imageSize is None:
            imageSize = img.shape[:2]
        else: 
            assert imageSize == img.shape[:2]

        res = detect(img, aruco_detector, tag_corners, expected_tags_calibrate)

        # If the result is not nothing then save the results
        if res:
            obj_temp, img_temp = res
            objectPoints.append(obj_temp)
            imagePoints.append(img_temp)
        
    # calibrateCamera expects (width, height) so reverse image_size
    imageSize = imageSize[::-1]

    # Calibrate the camera
    rms, cameraMatrix, distCoeffs, _, _ = cv.calibrateCamera(
        objectPoints,  # list of arrays of shape (4, 3) where the len(list) = # of images
        imagePoints,  # list of arrays of shape (4, 2) where the len(list) = # of images
        imageSize, # tuple (width, height)
        None, # cameraMatrix (not known)
        None, # distCoeffs (not known)
    )
    print("RMS error: ", rms)

    # cameraMatrix: intrinsic camera_matrix
    # distCoeffs: distortion coefficients, for radial and tangential distortion
    return cameraMatrix, distCoeffs

def detect(img, aruco_detector, tag_corners, expected_tags):

    # For each image, detect the ArUco tags using OpenCV's ArUco detector
    # Detect ArUco markers in an image
    # Returns: corners (list of numpy arrays), ids (numpy array)
    corners, ids, _ = aruco_detector.detectMarkers(img)

    # Check if any markers were detected
    if ids is not None:
        # Extract the corner coordinates from the detected tags
        # corners: list of length N (number of detected tags)
        #   - each element is a numpy array of shape (1, 4, 2) containing the 4 corner coordinates (x, y)
        # ids: numpy array of shape (N, 1) containing the tag IDs for each detected marker
        # Example: if 3 tags detected, corners will be a list of 3 arrays, ids will be shape (3, 1)
        obj_temp = []
        img_temp = []

        # Go through each corner and fill in obj_temp and img_temp 
        for i in range(len(corners)):
            
            marker_id = ids[i].squeeze() # Squeeze to convert from (1,) to scalars
            if marker_id not in expected_tags:
                continue

            # Save the expected position of the corners to obj_temp, if the number of tag_corners is 1 then it can only be that corner
            if len(tag_corners) == 1:
                expected_corners = tag_corners[0]
            else:
                expected_corners = tag_corners[marker_id]
            obj_temp.append(expected_corners)

            # Save the detected position of the corners to img_temp
            detected_corners = corners[i]
            img_temp.append(detected_corners.reshape(-1, 2).astype(np.float32)) # Reshape from (1, 4, 2) to (4, 2) and make sure its float32

        # If no valid tags were found then return
        if not obj_temp:
            return

        # Combine all the markers in the image
        return np.concatenate(obj_temp), np.concatenate(img_temp)

    else:
        # No tags detected in this image, skip it
        return


def estimate_pose(dir_path, cameraMatrix, distCoeffs, tag_corners, expected_tags_estimate, display=False):
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    aruco_params = cv.aruco.DetectorParameters()

    # Create the ArUco detector
    aruco_detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Each element of these lists represents one image
    imgs_rgb = []
    imgs_bgr = []
    c2ws = []

    # Loop through all your calibration images
    assert os.path.isdir(dir_path)
    for file_name in os.listdir(dir_path):
        
        file_path = dir_path + "/" + file_name
        img_rgb = cv.imread(file_path, cv.IMREAD_COLOR_RGB)
        img_bgr = cv.imread(file_path, cv.IMREAD_COLOR_BGR)
        assert img_bgr is not None

        # Convert to grayscale to enhance the detection process
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

        res = detect(img_gray, aruco_detector, tag_corners, expected_tags_estimate)

        # If the result is not nothing then try to the PnP problem
        if res:
            obj_temp, img_temp = res
            success, rvec, tvec = cv.solvePnP(obj_temp, img_temp, cameraMatrix, distCoeffs)

            if success:
                R, _ = cv.Rodrigues(rvec)
                C2W = np.eye(4)
                C2W[:3, :3] = R.T
                C2W[:3, 3] = -(R.T @ tvec).squeeze()
                
                imgs_rgb.append(img_rgb)
                imgs_bgr.append(img_bgr)
                c2ws.append(C2W)
        
    visualize(imgs_rgb, c2ws, cameraMatrix) if display else None
    return imgs_bgr, c2ws

def visualize(imgs, c2ws, cameraMatrix):
    server = viser.ViserServer(share=True)

    # Example of visualizing a camera frustum (in practice loop over all images)
    H, W = imgs[0].shape[:2]
    K = cameraMatrix

    for i in range(len(imgs)):
        img = imgs[i]

        # Resize the images so viser can render them
        while img.shape[0] * img.shape[1] >  2000 * 2000:
            img = cv.resize(img, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

        c2w = c2ws[i]

        # c2w is the camera-to-world transformation matrix (3x4), and K is the camera intrinsic matrix (3x3)
        server.scene.add_camera_frustum(
            f"/cameras/{i}", # give it a name
            fov=2 * np.arctan2(H / 2, K[1, 1]), # field of view
            aspect=W / H, # aspect ratio
            scale=0.06, # scale of the camera frustum change if too small/big
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, # orientation in quaternion format
            position=c2w[:3, 3], # position of the camera
            image=img
        )

    while True:
        time.sleep(0.1)  # Wait to allow visualization to run

def undistort_imgs(imgs, cameraMatrix, distCoeffs, display=False):
    undistorted_imgs = []

    for img in imgs:
        undistorted_img = cv.undistort(img, cameraMatrix, distCoeffs)
        display_img(undistorted_img, bgr=True) if display else None
        undistorted_imgs.append(undistorted_img)

    return undistorted_imgs

def look_at_origin(pos):
  # Camera looks towards the origin
  forward = -pos / np.linalg.norm(pos)  # Normalize the direction vector

  # Define up vector (assuming z-up)
  up = np.array([0, 0, 1])

  # Compute right vector using cross product
  right = np.cross(up, forward)
  right = right / np.linalg.norm(right)

  # Recompute up vector to ensure orthogonality
  up = np.cross(forward, right)

  # Create the camera-to-world matrix
  c2w = np.eye(4)
  c2w[:3, 0] = right
  c2w[:3, 1] = up
  c2w[:3, 2] = forward
  c2w[:3, 3] = pos

  return c2w

def rot_z(phi):
    return np.array([
        [ math.cos(phi), -math.sin(phi), 0, 0],
        [ math.sin(phi),  math.cos(phi), 0, 0],
        [ 0,              0,             1, 0],
        [ 0,              0,             0, 1],
    ])

def generate_extrinsics(c2w_train):
  START_POS = c2w_train[:3, 3]
  NUM_SAMPLES = 60

  extrinsics = []
  for phi in np.linspace(360., 0., NUM_SAMPLES, endpoint=False):
      c2w = look_at_origin(START_POS)
      extrinsic = rot_z(phi/180.*np.pi) @ c2w
      extrinsics.append(extrinsic)

  return extrinsics

def save_imgs(undistorted_imgs, cameraMatrix, c2ws, training_split, side_dim):
    N = len(undistorted_imgs)
    assert N == len(c2ws)

    # Get the scaling factor
    h, w = undistorted_imgs[0].shape[:2]
    scale_factor = side_dim / max(h, w)
     
    # Convert images to rgb for later in the pipeline and resize them
    for i in range(N):
        img = undistorted_imgs[i] 
        img = cv.resize(img, (0, 0), fx = scale_factor, fy = scale_factor, interpolation = cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        undistorted_imgs[i] = img

    # Training splits
    i_train = int(training_split * N)

    # Shuffle the data
    indices = np.random.permutation(N)
    undistorted_imgs = [undistorted_imgs[i] for i in indices]
    c2ws = [c2ws[i] for i in indices]

    # Use all of the data for training
    images_train = undistorted_imgs[:i_train]
    c2ws_train = c2ws[:i_train]

    # Use some of the data for validation
    images_val = undistorted_imgs[i_train:]
    c2ws_val = c2ws[i_train:]
    
    c2ws_test = generate_extrinsics(c2ws_train[0])

    # Calculating focal length (including the scale factor)
    fx = cameraMatrix[0, 0] * scale_factor
    fy = cameraMatrix[1, 1] * scale_factor
    focal = (fx + fy) / 2

    np.savez(
        './data/my_data.npz',
        images_train=images_train,    # (N_train, H, W, 3)
        c2ws_train=c2ws_train,        # (N_train, 4, 4)
        images_val=images_val,        # (N_val, H, W, 3)
        c2ws_val=c2ws_val,            # (N_val, 4, 4)
        c2ws_test=c2ws_test,          # (N_test, 4, 4)
        focal=focal                   # float
    )

if __name__ == "__main__":
    side_dim = 300

    expected_tags = [0, 1, 2, 3, 4, 5]

    tag_corners_lafufu = get_tag_corners(0.06, 3, 2, 0.09, 0.07567)
    # Since the A4 paper was printed on letter, the dimensions were measured manually
    tag_corners_custom = get_tag_corners(0.055, 3, 2, 0.0825, 0.06936) 

    # # Test Code for the Lafufu dataset
    # cameraMatrix, distCoeffs = calibrate("./data/example_imgs/calibrate", tag_corners_lafufu, expected_tags)
    # imgs, c2ws = estimate_pose("./data/example_imgs/lafufu", cameraMatrix, distCoeffs, tag_corners_lafufu, expected_tags, display=False)
    # undistorted_imgs = undistort_imgs(imgs, cameraMatrix, distCoeffs)
    # save_imgs(undistorted_imgs, cameraMatrix, c2ws, training_split=0.9, side_dim=side_dim)

    # Code for the custom dataset
    cameraMatrix, distCoeffs = calibrate("./data/custom_imgs/calibrate", tag_corners_custom, expected_tags)
    imgs, c2ws = estimate_pose("./data/custom_imgs/steve", cameraMatrix, distCoeffs, tag_corners_custom, expected_tags, display=False)
    undistorted_imgs = undistort_imgs(imgs, cameraMatrix, distCoeffs, display=False)
    save_imgs(undistorted_imgs, cameraMatrix, c2ws, training_split=1.0, side_dim=side_dim)