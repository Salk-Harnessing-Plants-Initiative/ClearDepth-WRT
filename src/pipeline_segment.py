# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:46:18 2023

@author: linwang
"""

# %% import library
import os, cv2, math, csv
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import warnings

warnings.filterwarnings("ignore")

import torch
import albumentations as album
import argparse

import segmentation_models_pytorch as smp

import segmentation_models_pytorch.utils


# %% add pading and crop images to patch size
def add_0padding_crop(
    patch_size,
    overlap_size,
    image_path,
    image_path_padding,
    image_path_crop,
    pad_image=False,
):
    """Add zero padding to the size of image and crop it for patch.

    Args:
        patch_size: expected patch size of deep learning model.
        overlap_size: the expected overlap/border of two adjacent images.
        image_path: the path where saved the original images.
        image_path_padding: the expected path to save padding images.
        image_path_crop: the expected path to save cropped images.
        pad_image: boolean data, where True means return padding images.

    Returns
        Add padding of current image and save padding images.
    """
    color = [0, 0, 0]  # add zero padding

    # get the image list from image_path
    image_list = [
        os.path.relpath(os.path.join(root, file), image_path)
        for root, _, files in os.walk(image_path)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    # print(f"image_path in add_0padding_crop: {image_path}")
    # print(f"image_list: {image_list}")

    for name in image_list:
        # pass
        im = cv2.imread(os.path.join(image_path, name))
        shape_0, shape_1 = im.shape[0], im.shape[1]
        n_0, n_1 = math.ceil(shape_0 / (patch_size - overlap_size / 2)), math.ceil(
            shape_1 / (patch_size - overlap_size / 2)
        )
        top, bottom = math.ceil(
            (n_0 * (patch_size - overlap_size / 2) - shape_0) / 2
        ), math.floor((n_0 * (patch_size - overlap_size / 2) - shape_0) / 2)
        left, right = math.ceil(
            (n_1 * (patch_size - overlap_size / 2) - shape_1) / 2
        ), math.floor((n_1 * (patch_size - overlap_size / 2) - shape_1) / 2)
        im_pad = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        if pad_image:
            name_pad = os.path.join(image_path_padding, name)
            cv2.imwrite(name_pad, im_pad)

        # create the crop folders
        crop_folder = os.path.join(image_path_crop, os.path.dirname(name))
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)

        idx = 0
        for i in range(n_0):
            for j in range(n_1):
                idx += 1
                crop_name = str(os.path.splitext(name)[0]) + "_" + str(idx) + ".png"
                top = i * (patch_size - overlap_size)
                left = j * (patch_size - overlap_size)
                im_crop = im_pad[top : top + patch_size, left : left + patch_size, :]
                name_crop = os.path.join(image_path_crop, crop_name)
                cv2.imwrite(name_crop, im_crop)


def crop_image(image, true_dimensions):
    return album.CenterCrop(p=1, height=true_dimensions[0], width=true_dimensions[1])(
        image=image
    )


# %% stitch
def stitch_crop_images(
    patch_size, overlap_size, original_image_path, image_path, stitch_path
):
    """Stitch the prediction in patch size.

    Args:
        patch_size: expected patch size of deep learning model.
        overlap_size: the expected overlap/border of two adjacent images.
        original_image_path: the path where store original images.
        image_path: the path where store prediction in patch size.
        stitch_path: the expected path where store the stitched predictions.


    Returns:
        Save stitched predictions.

    """
    PredNameList = [
        os.path.relpath(os.path.join(root, file), image_path)
        for root, _, files in os.walk(image_path)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    image_name = []
    for i in range(len(PredNameList)):
        file_path = os.path.dirname(PredNameList[i])
        filename = os.path.splitext(os.path.basename(PredNameList[i]))[0]
        filename2 = filename.rsplit("_", 1)
        image_name.append(os.path.join(file_path, filename2[0]))
    image_name = np.array(image_name)
    image_name_unique = np.unique(image_name)

    for name in image_name_unique:
        image_crop_name_list = [
            os.path.relpath(os.path.join(root, file), image_path)
            for root, _, files in os.walk(image_path)
            for file in files
            if (file.endswith(".PNG") or file.endswith(".png"))
            and file.startswith(str(name))
            and not file.startswith(".")
        ]
        image_crop_count = len(image_crop_name_list)

        # will need the original image shape
        original_imageList = [
            os.path.relpath(os.path.join(root, file), original_image_path)
            for root, _, files in os.walk(original_image_path)
            for file in files
            if (file.endswith(".PNG") or file.endswith(".png"))
            and not file.startswith(".")
        ]
        if ((name + ".PNG") or (name + ".png")) in original_imageList:
            im = cv2.imread(os.path.join(original_image_path, name + ".PNG"))
            if im is None:
                im = cv2.imread(os.path.join(original_image_path, name + ".png"))
            # print(os.path.join(original_image_path, name + ".PNG"))
            shape_0, shape_1, shape_2 = im.shape[0], im.shape[1], im.shape[2]
            n_0, n_1 = math.ceil(shape_0 / (patch_size - overlap_size / 2)), math.ceil(
                shape_1 / (patch_size - overlap_size / 2)
            )

            n_0_idx = []
            n_1_idx = []
            index = []
            ind = 0
            # index of row and column
            for i in range(n_0):
                for j in range(n_1):
                    ind += 1
                    n_0_idx.append(i)
                    n_1_idx.append(j)
                    index.append(ind)

            ind_df = pd.DataFrame(
                {"n_0_idx": n_0_idx, "n_1_idx": n_1_idx, "index": index}
            )
            ind_array = np.array(ind_df)

            # ind = 0
            im_stitch = np.zeros(
                [
                    int(n_0 * (patch_size - overlap_size / 2)),
                    int(n_1 * (patch_size - overlap_size / 2)),
                    shape_2,
                ]
            )
            for i in range(n_0):
                for j in range(n_1):
                    # ind += 1
                    top = i * (patch_size - overlap_size)
                    left = j * (patch_size - overlap_size)
                    ind = np.squeeze(
                        ind_array[
                            np.where((ind_array[:, 0] == i) & (ind_array[:, 1] == j)), 2
                        ]
                    )
                    im_pred_patch = cv2.imread(
                        image_path + name + "_" + str(ind) + ".png"
                    )
                    im_pred_patch_name = image_path + name + "_" + str(ind) + ".png"
                    # im_crop = im_pad[top:top+patch_size, left:left+patch_size, :]
                    if top == 0 and left == 0:
                        im_stitch[
                            top : top + patch_size, left : left + patch_size, :
                        ] = im_pred_patch
                    elif top == 0 and left > 0:
                        left_ind = np.squeeze(
                            ind_array[
                                np.where(
                                    (ind_array[:, 0] == i) & (ind_array[:, 1] == j - 1)
                                ),
                                2,
                            ]
                        )

                        im_left = cv2.imread(
                            image_path + name + "_" + str(left_ind) + ".png"
                        )

                        # get the overlap area (leaft side of im_pred_patch, right side of im_pred_patch_left)
                        im_pred_patch_left = im_pred_patch[
                            0:patch_size, 0:overlap_size, :
                        ]
                        im_left_right = im_left[0:patch_size, -overlap_size:, :]

                        # calculate maximum value of overlapping area
                        im_stitch[
                            top : top + patch_size, left : left + overlap_size, :
                        ] = np.maximum(im_pred_patch_left, im_left_right)
                        im_stitch[
                            top : top + patch_size,
                            left + overlap_size - 1 : left + patch_size - 1,
                            :,
                        ] = im_pred_patch[
                            0:patch_size, overlap_size - 1 : patch_size - 1, :
                        ]

                    elif top > 0 and left == 0:
                        top_ind = np.squeeze(
                            ind_array[
                                np.where(
                                    (ind_array[:, 0] == i - 1) & (ind_array[:, 1] == j)
                                ),
                                2,
                            ]
                        )

                        im_top = cv2.imread(
                            image_path + name + "_" + str(top_ind) + ".png"
                        )

                        # get the overlap area (top side of im_pred_patch, bottom side of im_pred_patch_top)
                        im_pred_patch_top = im_pred_patch[
                            0:overlap_size, 0:patch_size, :
                        ]
                        im_top_bottom = im_top[-overlap_size:, 0:patch_size, :]

                        # calculate maximum value of overlapping area
                        im_stitch[
                            top : top + overlap_size, left : left + patch_size, :
                        ] = np.maximum(im_pred_patch_top, im_top_bottom)
                        im_stitch[
                            top + overlap_size - 1 : top + patch_size - 1,
                            left : left + patch_size,
                            :,
                        ] = im_pred_patch[
                            overlap_size - 1 : patch_size - 1, 0:patch_size, :
                        ]
                    else:
                        top_ind = np.squeeze(
                            ind_array[
                                np.where(
                                    (ind_array[:, 0] == i - 1) & (ind_array[:, 1] == j)
                                ),
                                2,
                            ]
                        )
                        left_ind = np.squeeze(
                            ind_array[
                                np.where(
                                    (ind_array[:, 0] == i) & (ind_array[:, 1] == j - 1)
                                ),
                                2,
                            ]
                        )

                        im_top = cv2.imread(
                            image_path + name + "_" + str(top_ind) + ".png"
                        )
                        # get the overlap area (top side of im_pred_patch, bottom side of im_pred_patch_top)
                        im_pred_patch_top = im_pred_patch[
                            0:overlap_size, 0:patch_size, :
                        ]
                        im_top_bottom = im_top[-overlap_size:, 0:patch_size, :]

                        im_stitch[
                            top : top + overlap_size, left : left + patch_size, :
                        ] = np.maximum(im_pred_patch_top, im_top_bottom)
                        im_stitch[
                            top + overlap_size - 1 : top + patch_size - 1,
                            left : left + patch_size,
                            :,
                        ] = im_pred_patch[
                            overlap_size - 1 : patch_size - 1, 0:patch_size, :
                        ]

                        im_left = cv2.imread(
                            image_path + name + "_" + str(left_ind) + ".png"
                        )
                        # get the overlap area (leaft side of im_pred_patch, right side of im_pred_patch_left)
                        im_pred_patch_left = im_pred_patch[
                            0:patch_size, 0:overlap_size, :
                        ]
                        im_left_right = im_left[0:patch_size, -overlap_size:, :]

                        # calculate maximum value of overlapping area
                        im_stitch[
                            top : top + patch_size, left : left + overlap_size, :
                        ] = np.maximum(im_pred_patch_left, im_left_right)
                        im_stitch[
                            top : top + patch_size,
                            left + overlap_size - 1 : left + patch_size - 1,
                            :,
                        ] = im_pred_patch[
                            0:patch_size, overlap_size - 1 : patch_size - 1, :
                        ]
            print(f"stitching image: {name}")
            if "/" in name:
                stitch_folder = os.path.join(stitch_path, name.split("/")[:-1][0])
                if not os.path.exists(stitch_folder):
                    os.makedirs(stitch_folder)
                name_stitch = os.path.join(stitch_path, name + ".png")
            else:
                name_stitch = os.path.join(stitch_path, name + ".png")

            cv2.imwrite(name_stitch, im_stitch)


# %% remove boundary
def remove_boundary(seg_path, img_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    imageList = [
        os.path.relpath(os.path.join(root, file), img_path)
        for root, _, files in os.walk(img_path)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    print(f"remove boundary imageList: {imageList}")

    for i in range(len(imageList)):
        img_name = os.path.join(img_path, imageList[i])
        print(f"remove boundary img_name: {img_name}")
        image = cv2.imread(img_name)
        height, width = image.shape[0], image.shape[1]

        seg_name = os.path.join(seg_path, imageList[i])
        seg = cv2.imread(seg_name)
        height_seg, width_seg = seg.shape[0], seg.shape[1]

        crop_region = (
            int((width_seg - width) / 2),
            int((height_seg - height) / 2),
            width,
            height,
        )

        cropped_image = seg[
            crop_region[1] : crop_region[1] + crop_region[3],
            crop_region[0] : crop_region[0] + crop_region[2],
        ]

        savename_path = os.path.join(save_path, "Segmentation", imageList[i])
        save_dir = os.path.dirname(savename_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(savename_path, cropped_image)


# add first 1024 columns to the last
def add_left_to_right(img_folder):
    imgs = [
        os.path.relpath(os.path.join(root, file), img_folder)
        for root, _, files in os.walk(img_folder)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    new_folder = img_folder + "_extend"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for img in imgs:
        image = cv2.imread(os.path.join(img_folder, img))
        new_image = np.zeros((image.shape[0], image.shape[1] + 1024, 3))
        new_image[:, :-1024, :] = image
        new_image[:, -1024:, :] = image[:, :1024, :]

        savename_path = os.path.join(new_folder, img)
        save_dir = os.path.dirname(savename_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(savename_path, new_image)
        # return new_image


def remove_right(img_folder):
    imgs = [
        os.path.relpath(os.path.join(root, file), img_folder)
        for root, _, files in os.walk(img_folder)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]
    new_folder = img_folder + "_rmRight"

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for img in imgs:
        image = cv2.imread(os.path.join(img_folder, img))
        new_image = image[:, :-1024, :]

        savename_path = os.path.join(new_folder, img)
        save_dir = os.path.dirname(savename_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(savename_path, new_image)


# %% viz
# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


# %% prediction dataset
class PredictionDataset(torch.utils.data.Dataset):
    """Stanford Background Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        df (str): DataFrame containing images / labels paths
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
        self,
        df,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_paths = df["image_path"].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        # print(self.image_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        names = self.image_paths[i].rsplit("/", 1)[-1].split(".")[0]
        path_name = self.image_paths[i]

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample["image"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample["image"]

        return image, names, path_name

    def __len__(self):
        # return length of
        return len(self.image_paths)


def get_training_augmentation():
    train_transform = [
        # album.PadIfNeeded(min_height=550, min_width=660, always_apply=True, border_mode=0),
        # LW height and width change from 832 to 1000 to 1984 to 2720
        album.RandomCrop(height=1024, width=1024, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.5,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW size should be square and can be devided by 32
        # LW height and width change from 992 to 1120 to 2752
        album.PadIfNeeded(
            min_height=1024,
            min_width=1024,
            always_apply=True,
            border_mode=0,
            value=(0, 0, 0),
        ),
    ]
    return album.Compose(test_transform)


def get_test_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        # LW change min_height of 1120 to 6016 to 2016, same as val
        # LW change min_width of 992 to 4000 to 1120 for checking images instead of images_test
        album.PadIfNeeded(
            min_height=256, min_width=256, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


# %% main
def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Training Pipeline")
    parser.add_argument("--patch_size", default=1024, help="Cropped image patch size")
    parser.add_argument(
        "--overlap_size", default=256, help="Cropped images overlap pixels"
    )
    parser.add_argument("--image_path", required=True, help="Training images path")
    parser.add_argument("--stitch_path", required=True, help="Segmentation path")
    parser.add_argument("--model_name", required=True, help="Training model name")

    args = parser.parse_args()

    patch_size = int(args.patch_size)
    overlap_size = int(args.overlap_size)
    image_path = args.image_path
    stitch_path = args.stitch_path
    model_name = args.model_name

    #
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load best saved model checkpoint from the current run
    model_name_path = f"{model_name}.pth"
    print(f"model name: {model_name_path}")
    if os.path.exists(f"{model_name}.pth"):
        best_model = torch.load(f"{model_name}.pth", map_location=DEVICE)
        print("Loaded UNet model from this run.")
    else:
        raise ValueError("Model not available!")

    image_path_crop = os.path.join(stitch_path, "Image_Crop/")
    image_path_padding = os.path.join(stitch_path, "Image_Padding/")

    # pad_image = False
    # if pad_image:
    #     if not os.path.exists(image_path_padding):
    #         os.makedirs(image_path_padding)

    #     files_temp = os.listdir(image_path_padding)
    #     if len(files_temp) > 0:
    #         for items in files_temp:
    #             os.remove(os.path.join(image_path_padding, items))

    # get the image list from image_path
    image_list = [
        os.path.relpath(os.path.join(root, file), image_path)
        for root, _, files in os.walk(image_path)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    # crop images
    # image_path_crop = os.path.join(stitch_path, "Image_Crop/")
    if not os.path.exists(image_path_crop):
        os.makedirs(image_path_crop)

    files_temp = [
        os.path.relpath(os.path.join(root, file), image_path_crop)
        for root, _, files in os.walk(image_path_crop)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    # if len(files_temp) > 0:
    #     for items in files_temp:
    #         os.remove(os.path.join(image_path_crop, items))
    pad_image = False

    print(f"image_path: {image_path}")
    add_left_to_right(image_path)
    image_path_extend = image_path + "_extend"

    add_0padding_crop(
        patch_size,
        overlap_size,
        image_path_extend,
        image_path_padding,
        image_path_crop,
        pad_image,
    )
    print("Finished images cropping!")

    # generate metedata file
    subimage_list = [
        os.path.relpath(os.path.join(root, file), image_path_crop)
        for root, _, files in os.walk(image_path_crop)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    metadata_row = []
    for i in range(len(subimage_list)):
        image_path_i = image_path_crop + subimage_list[i]
        # label_path_i = label_path_crop+sublabel_list[i]
        label_path_i = image_path_crop + subimage_list[i]
        metadata_row.append([str(i + 1), image_path_i, label_path_i])

    metadata_file = "./metadata_tem.csv"

    header = ["image_id", "image_path", "label_colored_path"]
    with open(metadata_file, "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        writer.writerow([g for g in header])
        for x in range(len(metadata_row)):
            writer.writerow(metadata_row[x])

    # set up segmentation patch folder
    sample_preds_folder = os.path.join(stitch_path, "Segmentation_patch/")
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    # files = [
    #     os.path.relpath(os.path.join(root, file), sample_preds_folder)
    #     for root, _, files in os.walk(sample_preds_folder)
    #     for file in files
    #     if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    # ]

    # if len(files) > 0:
    #     for items in files:
    #         os.remove(os.path.join(sample_preds_folder, items))

    # setup model parameters
    ENCODER = "resnet101"
    ENCODER_WEIGHTS = "imagenet"
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # check the color
    class_dict = pd.read_csv("./label_class_dict_lr.csv")
    class_names = class_dict["name"].tolist()
    class_rgb_values = class_dict[["r", "g", "b"]].values.tolist()

    select_classes = ["background", "root"]
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # setup dataset
    metadata_df = pd.read_csv("metadata_tem.csv")
    test_dataset = PredictionDataset(
        metadata_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    test_dataset_vis = PredictionDataset(
        metadata_df,
        class_rgb_values=select_class_rgb_values,
    )

    print(len(test_dataset))
    # predict patch segmentation
    for idx in range(len(test_dataset)):  # len(test_dataset)
        image, names, path_name = test_dataset[idx]
        print(
            f"Predicting {idx+1}th image among {len(test_dataset)} images: {path_name}"
        )
        image_vis = test_dataset_vis[idx][0].astype("uint8")
        true_dimensions = image_vis.shape
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to foreground
        # pred_foreground_heatmap = crop_image(pred_mask[:,:,select_classes.index('root')], true_dimensions)['image']
        pred_mask = crop_image(
            colour_code_segmentation(
                reverse_one_hot(pred_mask), select_class_rgb_values
            ),
            true_dimensions,
        )["image"]
        # get the subfolder name
        path_parts = path_name.split("/")
        index = path_parts.index("Image_Crop")
        if index < len(path_parts) - 2:  # no subfolder after crop
            subfolder = path_parts[index + 1]
            save_path = os.path.join(sample_preds_folder, subfolder)
        else:
            save_path = sample_preds_folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, f"{names}.png"), pred_mask)
    print("Finish prediction!")

    # stitch
    original_image_path = image_path_extend
    image_path_t = sample_preds_folder
    stitch_path_b = os.path.join(stitch_path, "Segmentation_temp/")
    if not os.path.exists(stitch_path_b):
        os.makedirs(stitch_path_b)

    files = [
        os.path.relpath(os.path.join(root, file), stitch_path_b)
        for root, _, files in os.walk(stitch_path_b)
        for file in files
        if (file.endswith(".PNG") or file.endswith(".png")) and not file.startswith(".")
    ]

    if len(files) > 0:
        for items in files:
            os.remove(os.path.join(stitch_path_b, items))
    stitch_crop_images(
        patch_size, overlap_size, original_image_path, image_path_t, stitch_path_b
    )

    # remove boundary
    seg_path = stitch_path_b
    img_path = image_path_extend
    save_path = stitch_path
    remove_boundary(seg_path, img_path, save_path)
    remove_right(os.path.join(save_path, "Segmentation"))


if __name__ == "__main__":
    main()
