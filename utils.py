import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np

# Splits a filename of type "/path.../name.extension" and returns name
def path_to_filename(filename):
    return filename.split("/")[-1].split(".")[0]

# Loads an image from a specified path with a specified cv_type and displays it if the display flag is set to True
# Also converts to float is the to_float flag is set
def load_img(path, cv_type, to_float, display, shrink, size=1000):
    img = cv.imread(path, cv_type)
    assert img is not None

    if shrink:
        while img.shape[0] * img.shape[1] >  size * size:
            img = cv.resize(img, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    # Convert image front uints to floats for internal computation
    if to_float:
        img = uint8_to_float64(img) 

    bgr = True if cv_type == cv.IMREAD_COLOR_BGR else False
    display_img(img, bgr=bgr) if display else None

    img_name = path_to_filename(path)
    return img, img_name

# Displays the full image without it going out of the screen size, press a key to close the image
def display_img(img, bgr=True):

    # Ensure the image is in uint8's so that cvtColor works
    img = float64_to_uint8(img)

    # If the image is not grayscale then convert it to RGBA
    if img.ndim == 3: 
        if bgr: 
            if img.shape[2] == 4:
                img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
            else:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
        else: 
            if img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    ax.axis("on")

    # Close on key press
    def on_key(_):
        plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# Saves an image to a folder and creates the folder if it doesn't exist
def save_img(folder_path, file_name, img, normalize, bgr=True):
    if normalize:
        img = normalize_img(img)

    # Convert the image to uint8 for opencv
    if img.dtype == np.float64: 
        img = float64_to_uint8(img)

    # If the image is not bgr and not gray
    if not bgr and img.ndim == 3: 

        # If the image includes an alpha channel
        if img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)

        # If the image doesn't include an alpha channel
        else:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    os.makedirs(folder_path, exist_ok=True)                  # Make the folder in case it doesn't exist             
    success = cv.imwrite(folder_path + "/" + file_name + ".png", img) # Write to the folder and assert that it was successful
    assert success

def float64_to_uint8(img):
    return (img * 255).astype(np.uint8)

def uint8_to_float64(img):
    return img.astype(np.float64) / 255.0

# Normalizes an image to be in the range of [0, 1]
# Could be a grayscale image or a color image
def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())

# Given the image, and points, draws on a copy of the image and returns the mutated copy
# Expects img_pts to be an (nx2) matrix with (x, y) format
def draw_on_img(img, img_pts, radii, color):
    # Draw points on a copy of the image (so not to mutate the original)
    img_copy = img.copy()
    for pt in img_pts:
        x, y = int(pt[0]), int(pt[1])
        cv.circle(img_copy, (x, y), radius=radii, color=color, thickness=-1)
    return img_copy