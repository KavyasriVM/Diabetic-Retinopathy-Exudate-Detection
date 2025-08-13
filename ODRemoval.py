import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
from scipy.ndimage import binary_fill_holes

def detect_and_remove_optic_disc(image_path, margin_size=10, display_results=True):
    """
    Complete pipeline to detect and remove (blacken) optic disc from fundus images

    Parameters:
    - image_path: Path to the fundus image
    - margin_size: Extra margin to add around optic disc (in pixels)
    - display_results: Whether to display visualization of results

    Returns:
    - Dictionary containing processed images and detection info
    """
    # STEP 1: Read and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None

    # Store original for display
    orig_img = img.copy()
    img = cv2.resize(img, (512, 512))

    # STEP 2: Determine image type (gray or color)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)

    if mean_saturation < 50:
        image_type = "GRAY"
    else:
        image_type = "COLOR"

    # STEP 3: Channel selection and enhancement based on image type
    if image_type == "GRAY":
        # Use multichannel approach for gray images
        b, g, r = cv2.split(img)
        combined = cv2.addWeighted(cv2.addWeighted(r, 0.4, g, 0.4, 0), 0.8, b, 0.2, 0)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(combined)

        # Apply bilateral filtering for gray images
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # For gray images, also apply additional contrast enhancement
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    else:  # COLOR images
        # Use green-red combination for color images
        _, g, r = cv2.split(img)
        combined = cv2.addWeighted(r, 0.6, g, 0.4, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(combined)

    # STEP 4: Optic disc localization
    # Apply Gaussian blur for localization
    blurred = cv2.GaussianBlur(enhanced, (31, 31), 0)

    # Create anatomical prior (higher weights for regions where OD typically appears)
    h, w = enhanced.shape
    prior_map = np.ones((h, w), dtype=np.float32)

    # For gray images, emphasize upper left quadrant where OD often appears
    if image_type == "GRAY":
        # Higher weight for upper left quadrant
        prior_map[:h//2, :w//2] *= 2.0

        # Apply prior to blurred image
        blurred = blurred * prior_map

    # Find brightest region (likely optic disc center)
    od_y, od_x = np.unravel_index(np.argmax(blurred), blurred.shape)

    # STEP 5: Determine appropriate optic disc radius
    if image_type == "GRAY":
        od_radius = int(img.shape[0] * 0.07)  # Smaller radius for gray images
    else:
        od_radius = int(img.shape[0] * 0.08)  # Larger radius for color images

    # STEP 6: Create initial circular mask
    od_mask = np.zeros_like(enhanced)
    cv2.circle(od_mask, (od_x, od_y), od_radius, 255, -1)

    # Apply morphological operations to smooth mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    od_mask = cv2.morphologyEx(od_mask, cv2.MORPH_CLOSE, kernel)
    od_mask = binary_fill_holes(od_mask).astype(np.uint8) * 255

    # STEP 7: Create expanded mask with safety margin
    # Create structuring element for dilation
    margin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (margin_size*2+1, margin_size*2+1))

    # Dilate the mask to create extra margin
    expanded_mask = cv2.dilate(od_mask, margin_kernel, iterations=1)

    # STEP 8: Resize masks to match original image size
    orig_h, orig_w = orig_img.shape[:2]
    od_mask_resized = cv2.resize(od_mask, (orig_w, orig_h))
    expanded_mask_resized = cv2.resize(expanded_mask, (orig_w, orig_h))

    # STEP 9: Remove (blacken) optic disc in original image
    # Invert the expanded mask
    removal_mask = cv2.bitwise_not(expanded_mask_resized)

    # Apply mask to original image
    result_img = cv2.bitwise_and(orig_img, orig_img, mask=removal_mask)

    # Prepare overlay image for visualization
    overlay_img = orig_img.copy()
    overlay_mask = np.zeros_like(orig_img)
    overlay_mask[:, :, 0] = od_mask_resized  # Red channel only

    # Add overlay to original image
    alpha = 0.4
    overlay_result = cv2.addWeighted(overlay_img, 1, overlay_mask, alpha, 0)

    # Display results if requested
    if display_results:
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 5, 2)
        plt.imshow(od_mask_resized, cmap='gray')
        plt.title('Optic Disc Mask')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        plt.imshow(expanded_mask_resized, cmap='gray')
        plt.title(f'Expanded Mask (+{margin_size}px)')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        plt.imshow(cv2.cvtColor(overlay_result, cv2.COLOR_BGR2RGB))
        plt.title('Overlay of Mask on Original')
        plt.axis('off')

        plt.subplot(1, 5, 5)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Image with OD Removed')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Return results
    results = {
        'original_image': orig_img,
        'od_mask': od_mask_resized,
        'expanded_mask': expanded_mask_resized,
        'overlay_image': overlay_result,
        'od_removed_image': result_img,
        'detection_info': {
            'image_type': image_type,
            'od_center': (od_x * orig_w // 512, od_y * orig_h // 512),
            'od_radius': od_radius * orig_w // 512,
            'margin_size': margin_size
        }
    }

    return results

def batch_process_images(image_folder, output_folder, margin_size=10):
    """
    Process multiple images and save results

    Parameters:
    - image_folder: Path to folder containing retinal images
    - output_folder: Path to save processed images
    - margin_size: Extra margin to add around optic disc (in pixels)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, 'masks'))
        os.makedirs(os.path.join(output_folder, 'expanded_masks'))
        os.makedirs(os.path.join(output_folder, 'removed_od'))

    # Get all image files
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

    for i, filename in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {filename}")

        # Full path to image
        image_path = os.path.join(image_folder, filename)

        try:
            # Process image
            results = detect_and_remove_optic_disc(image_path, margin_size, display_results=False)

            if results is None:
                print(f"  Failed to process {filename}")
                continue

            # Save results
            base_name = os.path.splitext(filename)[0]

            # Save mask
            cv2.imwrite(os.path.join(output_folder, 'masks', f"{base_name}_od_mask.png"),
                       results['od_mask'])

            # Save expanded mask
            cv2.imwrite(os.path.join(output_folder, 'expanded_masks', f"{base_name}_expanded_mask.png"),
                       results['expanded_mask'])

            # Save image with OD removed
            cv2.imwrite(os.path.join(output_folder, 'removed_od', f"{base_name}_od_removed.png"),
                       results['od_removed_image'])

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")

    print(f"Processing complete. Results saved to {output_folder}")

# Example usage
if _name_ == "_main_":
    # Single image processing
    #image_path = "/content/drive/MyDrive/Aptos/Train_3/e19936582c61aug28.png"  # Change to your image path
    #results = detect_and_remove_optic_disc(image_path, margin_size=19)

    # To process an entire folder of images, uncomment these lines:
    image_folder = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_0"
    output_folder = "/content/drive/MyDrive/Retinopathy_Train_Segregated/ODremoved_Grade_0"
    batch_process_images(image_folder, output_folder, margin_size=25)

    image_folder1 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_3"
    output_folder1 = "/content/drive/MyDrive/Retinopathy_Train_Segregated/ODremoved_Grade_3"
    batch_process_images(image_folder1, output_folder1, margin_size=25)
