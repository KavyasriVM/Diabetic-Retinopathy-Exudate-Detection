import cv2
import numpy as np
import os
from tqdm import tqdm

def clahe_and_normalize(image):
    """
    Applies CLAHE and normalization to the green channel of a retinal image.

    Parameters:
    image (numpy.ndarray): Input image (BGR or grayscale).

    Returns:
    numpy.ndarray: CLAHE enhanced and normalized image.
    """
    # If BGR image, take the green channel
    if len(image.shape) == 3:
        green_channel = image[:, :, 1]
    else:
        green_channel = image

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_clahe = clahe.apply(green_channel)

    # Background estimation with heavy Gaussian blur
    bg_estimate = cv2.GaussianBlur(green_clahe, (99, 99), 0)

    # Subtract background
    foreground = cv2.subtract(green_clahe, bg_estimate)

    # Normalize to [0, 255]
    min_val = np.min(foreground)
    max_val = np.max(foreground)
    if max_val > min_val:
        normalized = np.uint8(255 * ((foreground - min_val) / (max_val - min_val)))
    else:
        normalized = foreground

    return normalized

def process_batch(input_folder, output_folder):
    """
    Processes a batch of images with CLAHE and normalization.

    Parameters:
    input_folder (str): Path to folder containing input images.
    output_folder (str): Path to save processed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue

        processed = clahe_and_normalize(img)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed)

# Example usage
if _name_ == "_main_":

    input_folder1 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_0_Aug"          # Folder with your original images
    output_folder1 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_0_preprocessed" # Folder where processed images will be saved

    process_batch(input_folder1, output_folder1)

    input_folder2 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_3_Aug"          # Folder with your original images
    output_folder2 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_3_preprocessed" # Folder where processed images will be saved

    process_batch(input_folder2, output_folder2)
