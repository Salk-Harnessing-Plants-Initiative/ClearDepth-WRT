import os
import cv2
import numpy as np
import pandas as pd


def main():
    seg_folder = "../Segmentation_v10_Jul8/Segmentation_rmRight"
    image_list = sorted([file for file in os.listdir(seg_folder)])

    counts = pd.DataFrame()
    for i, image_name in enumerate(image_list):
        print(f"{i+1}th image among {len(image_list)} images, image: {image_name}")
        image = cv2.imread(os.path.join(seg_folder, image_name))[:, :, 0]

        area = np.count_nonzero(image)

        new_data = pd.DataFrame([{"image_name": image_name, "root_pixel_count": area}])
        counts = pd.concat([counts, new_data], ignore_index=True)
    counts.to_csv("../Segmentation_v10_Jul8/root_pixel_counts.csv", index=False)


if __name__ == "__main__":
    main()
