import os
import cv2
import albumentations as A

# Set input/output paths (update if needed)
healthy_dir = '/content/drive/MyDrive/Retinopathy_Train_Segregated/ODremoved_Grade_0/removed_od'
exudate_dir = '/content/drive/MyDrive/Retinopathy_Train_Segregated/ODremoved_Grade_3/removed_od'


aug_healthy_dir = '/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_0_Aug'
aug_exudate_dir = '/content/drive/MyDrive/Retinopathy_Train_Segregated/Grade_3_Aug'


# Create output directories
os.makedirs(aug_healthy_dir, exist_ok=True)
os.makedirs(aug_exudate_dir, exist_ok=True)


# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussianBlur(p=0.2),
    A.ElasticTransform(p=0.2),
])

def augment_images(input_folder, output_folder, label='img'):
    image_files = sorted(os.listdir(input_folder))
    count = 0

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        for i in range(30):  # 20 augmentations per image
            augmented = transform(image=image)['image']
            aug_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.png"
            cv2.imwrite(os.path.join(output_folder, aug_name), augmented)
            count += 1

    print(f"Done augmenting {count} images in {output_folder}")

# Run augmentation for each folder
augment_images(healthy_dir, aug_healthy_dir, label='healthy')
augment_images(exudate_dir, aug_exudate_dir, label='exudate')

