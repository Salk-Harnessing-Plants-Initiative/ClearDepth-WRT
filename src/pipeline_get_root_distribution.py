import os
import pandas as pd
import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler


def func(x, beta):
    """D95 fitting model
    refer to paper: Jackson, R. B., Canadell, J., Ehleringer, J. R., Mooney, H. A.,
    Sala, O. E., & Schulze, E. D. (1996).
    A global analysis of root distributions for terrestrial biomes.
    Oecologia, 108, 389-411.
    """
    return 1 - np.power(beta, x)


def d95_model(nlayer, im_depth, data_pts):
    """Simulate and output the D95 model.

    Args:
        nlayer: the layers to seperate the segmentation images on verticle direction.
        im_depth: depth of the image.
        data_pts: Root pixel locations as an array of shape (n, 2).

    Returns:
        d95 variables:
            beta, beta value of the fitteed curve
            r2, fittness of the curve
            d95_layer, the layer index which greater than 95th percentile of root pixels
    """
    if len(data_pts) > 0:
        # get depth of each layer
        depth_layer = int(im_depth / nlayer)
        # get 95% root pixel number
        # count_all_95 = len(data_pts) * 0.95

        # get y locations of root pixels
        data_pts_y = data_pts[:, 1]

        # get the pixel counts for each layer and accumulated counts for each layer
        count_layer = []
        count_sum_layer = []
        # calculate the pixel number of root for each layer
        for j in range(nlayer):
            # get the location of each layer
            lower_bound = j * depth_layer
            upper_bound = (j + 1) * depth_layer
            pixel_per_layer = data_pts_y[
                (data_pts_y >= lower_bound) & (data_pts_y < upper_bound)
            ]
            # get count per layer
            count_per_layer = len(pixel_per_layer)
            count_layer.append(count_per_layer)

            # get accumulated count per layer
            count_layer_sum = sum(count_layer)
            count_sum_layer = np.append(count_sum_layer, count_layer_sum)

        # get accumulated frequency for each layer
        count_layer_sum_frac = count_sum_layer / len(data_pts)

        # get x and y for the power function to calculate beta and r2
        y = count_layer_sum_frac
        x = np.array(list(range(1, nlayer + 1)))

        popt, pcov = curve_fit(func, x, y)
        beta = np.squeeze(popt)
        y_esti = 1 - np.power(popt, x)

        corr_matrix = np.corrcoef(y, y_esti)
        corr = corr_matrix[0, 1]
        r2 = corr**2

        # get 95th layer
        d95_layer = np.where(count_layer_sum_frac > 0.95)[0][0]
        # # get d95_variables for count_layer_sum_frac of each layer
        # d95_variables = {}

        # # Iterate over the array and assign each value to a dictionary key
        # for i, value in enumerate(count_layer_sum_frac):
        #     var_name = f"count_accu_frac_layer_{i+1}"  # Create a variable name
        #     d95_variables[var_name] = value

    else:
        beta = r2 = d95_layer = np.nan
        # # get d95_variables for count_layer_sum_frac of each layer
        # d95_variables = {}

        # # Iterate over the array and assign each value to a dictionary key
        # count_layer_sum_frac = np.full(nlayer, np.nan)
        # for i, value in enumerate(count_layer_sum_frac):
        #     var_name = f"count_accu_frac_layer_{i+1}"  # Create a variable name
        #     d95_variables[var_name] = value

    return beta, r2, d95_layer


def get_traits(seg_pot):
    """Get traits based on the tip locations.

    Args:
        seg_pot: root segmentation for root within pot.

    Returns:
        A dataframe with the tip-based traits of rach image or plant.
            counts: tip number counts of each image or plant.
            layerX_counts: the tip counts of different layers, layer 1 is bottom, layer 5 is top.
            p95: the 95th percentile of the tip y locations from bottom, so it's 5th percentile of the actual tip_y_cm value.
            stdx, stdy: standard deviations of x- and y- axis.
    """

    # get seg_pot shape
    height = seg_pot.shape[0]
    layer_height = int(height / 17)
    # get total root area of each plant and distribution of 17 layers
    value, count = np.unique(seg_pot, return_counts=True)
    areas_layer = {}
    if len(count) > 1:
        total_root_area = count[1]
        for i in range(17):
            seg_layer = seg_pot[layer_height * i : layer_height * (i + 1)]
            _, count_layer = np.unique(seg_layer, return_counts=True)
            area_name = f"root_area_layer_{i+1}"  # Create a variable name
            ratio_name = f"root_area_ratio_layer_{i+1}"
            if len(count_layer) > 1:
                root_area_layer = count_layer[1]
                areas_layer[area_name] = root_area_layer
                areas_layer[ratio_name] = root_area_layer / total_root_area
            else:
                root_area_layer = 0
                areas_layer[area_name] = 0
                areas_layer[ratio_name] = 0

    else:
        total_root_area = 0
        for i in range(17):
            area_name = f"root_area_layer_{i+1}"  # Create a variable name
            ratio_name = f"root_area_ratio_layer_{i+1}"
            areas_layer[area_name] = 0
            areas_layer[ratio_name] = 0
    # D95 model
    nlayer = 17
    im_depth = seg_pot.shape[0]
    # get data points (n,2) of x and y locations
    y_indices, x_indices = np.nonzero(seg_pot)
    data_pts = np.column_stack((x_indices, y_indices))
    beta, r2, d95_layer = d95_model(nlayer, im_depth, data_pts)

    return total_root_area, areas_layer, beta, r2, d95_layer


def get_depth_dist(img):
    root_px = np.where(img > 0)  # root_px[0] is verticle axis (y)
    height = img.shape[0]
    scale = height / 17
    if len(root_px[0]) > 0:
        root_y_min = np.min(root_px[0] / scale)
        root_y_max = np.max(root_px[0] / scale)
        root_y_std = np.std(root_px[0] / scale)

        root_y_mean = np.mean(root_px[0] / scale)
        root_y_median = np.median(root_px[0] / scale)
        root_y_p5 = np.percentile(root_px[0] / scale, 5)
        root_y_p25 = np.percentile(root_px[0] / scale, 25)
        root_y_p75 = np.percentile(root_px[0] / scale, 75)
        root_y_p95 = np.percentile(root_px[0] / scale, 95)

        # get the normalized summary based on the height of plant
        depth = root_y_max - root_y_min
        root_y_mean_norm = (root_y_mean - root_y_min) / depth
        root_y_median_norm = (root_y_median - root_y_min) / depth
        root_y_p5_norm = (root_y_p5 - root_y_min) / depth
        root_y_p25_norm = (root_y_p25 - root_y_min) / depth
        root_y_p75_norm = (root_y_p75 - root_y_min) / depth
        root_y_p95_norm = (root_y_p95 - root_y_min) / depth

    else:
        root_y_min = np.nan
        root_y_max = np.nan
        root_y_std = np.nan
        root_y_mean = np.nan
        root_y_median = np.nan
        root_y_p5 = np.nan
        root_y_p25 = np.nan
        root_y_p75 = np.nan
        root_y_p95 = np.nan
        root_y_mean_norm = np.nan
        root_y_median_norm = np.nan
        root_y_p5_norm = np.nan
        root_y_p25_norm = np.nan
        root_y_p75_norm = np.nan
        root_y_p95_norm = np.nan

    return (
        root_y_min,
        root_y_max,
        root_y_std,
        root_y_mean,
        root_y_median,
        root_y_p5,
        root_y_p25,
        root_y_p75,
        root_y_p95,
        root_y_mean_norm,
        root_y_median_norm,
        root_y_p5_norm,
        root_y_p25_norm,
        root_y_p75_norm,
        root_y_p95_norm,
    )


def get_root_distribution(seg_path):
    seg_names = [file for file in os.listdir(seg_path)]
    # print(f"seg_names: {seg_names}")
    root_distribution = pd.DataFrame()
    for seg_name in seg_names:
        seg = cv2.imread(os.path.join(seg_path, seg_name))[:, :, 0]
        value, count = np.unique(seg, return_counts=True)
        print(f"seg_name: {seg_name}, seg shape: {seg.shape}, count: {count}")
        # seperate as 17 layers = 17 cm
        # top start from 140, bottom end at 4458
        # top start from 270, bottom end at 4350 to exclude the top and bottom clearpot
        # get the seg_pot
        seg_pot = seg[270:4350, :]
        total_root_area, areas_layer, beta, r2, d95_layer = get_traits(seg_pot)

        (
            root_y_min,
            root_y_max,
            root_y_std,
            root_y_mean,
            root_y_median,
            root_y_p5,
            root_y_p25,
            root_y_p75,
            root_y_p95,
            root_y_mean_norm,
            root_y_median_norm,
            root_y_p5_norm,
            root_y_p25_norm,
            root_y_p75_norm,
            root_y_p95_norm,
        ) = get_depth_dist(seg_pot)
        data_new = pd.DataFrame(
            [
                {
                    "image_name": seg_name,
                    "total_root_area": total_root_area,
                    "beta": beta,
                    "r2": r2,
                    "d95_layer": d95_layer,
                    "root_y_min": root_y_min,
                    "root_y_max": root_y_max,
                    "root_y_std": root_y_std,
                    "root_y_mean": root_y_mean,
                    "root_y_median": root_y_median,
                    "root_y_p5": root_y_p5,
                    "root_y_p25": root_y_p25,
                    "root_y_p75": root_y_p75,
                    "root_y_p95": root_y_p95,
                    "root_y_mean_norm": root_y_mean_norm,
                    "root_y_median_norm": root_y_median_norm,
                    "root_y_p5_norm": root_y_p5_norm,
                    "root_y_p25_norm": root_y_p25_norm,
                    "root_y_p75_norm": root_y_p75_norm,
                    "root_y_p95_norm": root_y_p95_norm,
                    **areas_layer,
                }
            ]
        )
        root_distribution = pd.concat([root_distribution, data_new])
        root_distribution = root_distribution.reset_index(drop=True)
    print(f"root_distribution: {root_distribution}")
    return root_distribution


def kmeans_cluster(data):
    # normalize data using Z-score normalization
    scaler = StandardScaler()
    print(f"data type: {type(data)}")
    if type(data) == pd.Series:
        data_copy = np.array(data).reshape(-1, 1)
        data_scaled = scaler.fit_transform(data_copy)
        data_scaled = pd.DataFrame(data_scaled)
    else:
        data_scaled = scaler.fit_transform(data)

    # Perform k-means clustering
    num_clusters = 10  # adjust this based on your requirements
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    data["cluster"] = kmeans.fit_predict(data_scaled)
    data = data.sort_values(by="cluster")

    # data = pd.concat([data, data_ori["genotype"]], axis=1)
    return data


def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Training Pipeline")
    parser.add_argument("--seg_path", required=True, help="Segmentation images path")
    parser.add_argument(
        "--save_path", required=True, help="Tip distribution save folder"
    )
    args = parser.parse_args()

    seg_path = args.seg_path
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    root_distribution = get_root_distribution(seg_path)
    root_distribution.to_csv(
        os.path.join(save_path, "root_distribution.csv"), index=False
    )
    # root_distribution = pd.read_csv(os.path.join(save_path, "root_distribution.csv"))

    # perform cluster only with total area
    data_cluster = kmeans_cluster(root_distribution.iloc[:, 1])
    data_cluster.insert(0, "image_name", root_distribution.iloc[:, 0])
    # print(f"cluster: {data_cluster}")
    data_cluster.to_csv(os.path.join(save_path, "cluster_total_area.csv"), index=False)

    # perform cluster only with total area and area in layers
    data = pd.concat(
        [
            root_distribution.iloc[:, 1],
            root_distribution.filter(like="root_area_layer", axis=1),
        ],
        axis=1,
    )
    data_cluster = kmeans_cluster(data)
    data_cluster.insert(0, "image_name", root_distribution.iloc[:, 0])
    # print(f"cluster: {data_cluster}")
    data_cluster.to_csv(
        os.path.join(save_path, "cluster_total_area_and_layer_area.csv"), index=False
    )

    # perform cluster only with total area and area in layers
    data = pd.concat(
        [
            root_distribution.iloc[:, 1],
            root_distribution.filter(like="root_area_ratio_layer", axis=1),
        ],
        axis=1,
    )
    data_cluster = kmeans_cluster(data)
    data_cluster.insert(0, "image_name", root_distribution.iloc[:, 0])
    # print(f"cluster: {data_cluster}")
    data_cluster.to_csv(
        os.path.join(save_path, "cluster_total_area_and_layer_ratio.csv"), index=False
    )

    # perform cluster with all traits
    data = root_distribution.iloc[:, 1:]
    data_cluster = kmeans_cluster(data)
    data_cluster.insert(0, "image_name", root_distribution.iloc[:, 0])
    # print(f"cluster: {data_cluster}")
    data_cluster.to_csv(os.path.join(save_path, "cluster_all_traits.csv"), index=False)

    # tip_loc = pd.read_csv(tip_csv_path)


if __name__ == "__main__":
    main()
