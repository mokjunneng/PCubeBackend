from itertools import combinations
import math
import time

from sklearn.cluster import KMeans
import numpy as np

def kmeans_clustering(img, k=2):
    kmeans = KMeans(n_clusters=k).fit(img.flatten().reshape(-1, 1))
    cluster_centers = kmeans.cluster_centers_.squeeze(1)
    labels = kmeans.labels_
    labels = [cluster_centers[label] for label in labels]
    clustered_img = np.array(labels).reshape(img.shape)
    return clustered_img, cluster_centers, labels

def kmeans_mass_function(datapoints):
    kmeans_since = time.time()
    clustered_points, cluster_centers, labels = kmeans_clustering(datapoints)
    kmeans_duration = time.time() - kmeans_since
    print(f"Kmeans clustering ran for {kmeans_duration}s")

    gaussian_since = time.time()
    gaussian_datapoints_for_all_subset_clusters, cluster_set = convert_to_gaussian_for_all_subsets_in_clusters(datapoints, labels, cluster_centers)
    gaussian_conversion_duration = time.time() - gaussian_since
    print(f"Gaussian conversion ran for {gaussian_conversion_duration}s")
    #TODO: Do we really want to output the frame of discernment along with the mass function?
    return gaussian_datapoints_for_all_subset_clusters, cluster_set

def convert_to_gaussian_for_all_subsets_in_clusters(img, img_labels, clusters):
    '''
    Args:
        img: original image
        img_labels: image pixel elements are cluster index
        clusters: clusters of interest to convert to gaussian
    Returns:
        gaussian_img_for_all_subsets_windowed
        cluster_set: A.K.A frame of discernments
    '''
    # Combinations of cluster in clusters from i=1 to len(clusters)
    cluster_set = _set_combinations(clusters)

    # TODO: Is there a better way to index the clusters
    # img_labels = convert_labels2idx(img_labels, clusters)

    sigmas = {}
    means = {}
    gaussian_img_for_all_subsets = []
    for cs in cluster_set:
        gaussian_img, sigmas, means = convert_to_gaussian_for_subset(img.flatten().reshape(-1, 1), img_labels, cs, sigmas, means)
        gaussian_img_for_all_subsets.append(gaussian_img.reshape(img.shape))
    # TODO: separate window mean function?
    gaussian_img_for_all_subsets_windowed = [apply_window_mean(gaussian_img).reshape(-1, 1) for gaussian_img in gaussian_img_for_all_subsets]
    gaussian_img_for_all_subsets_windowed = np.stack(gaussian_img_for_all_subsets_windowed)
    return gaussian_img_for_all_subsets_windowed, cluster_set

def convert_to_gaussian_for_subset(img, img_labels, subset, sigmas, means):
    for s in subset:
        if sigmas.get(s) and means.get(s):
            mean = means.get(s)
            sigma = sigmas.get(s)
        else:
            img_s = get_pixels_from_img_with_index_s(img, img_labels, s)
            mean = np.mean(img_s)
            sigma = _standard_deviation(img_s, mean)
            means[s] = mean
            sigmas[s] = sigma
    means_for_compute = [means[s] for s in subset]
    sigmas_for_compute = [sigmas[s] for s in subset]
    img_gaussian = compute_gaussian_for_subset(img, means_for_compute, sigmas_for_compute)
    return img_gaussian, sigmas, means

def get_pixels_from_img_with_index_s(img, img_labels, idx):
    indices = np.where(img_labels != idx)
    pixels_with_index_s = np.delete(img, indices)
    return pixels_with_index_s

def compute_gaussian_for_subset(img, means, sigmas):
    subset_mean = np.mean(means)
    subset_sigma = np.max(sigmas)
    img_gaussian = gaussian_eqn(img, subset_mean, subset_sigma)
    return img_gaussian

def gaussian_eqn(datapoints, mean, sigma):
    gaussian = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-(datapoints - mean)**2 / (2*sigma**2))
    return gaussian

def apply_window_mean(img, window_size=3):
    offset = int((window_size - 1) / 2)
    H, W = img.shape
    H_prime, W_prime = H - 2*offset, W - 2*offset
    new_img = np.empty((H_prime, W_prime))
    for h in range(H_prime):
        for w in range(W_prime):
            window = img[h:h+window_size, w:w+window_size]
            new_img[h, w] = np.sum(window) / window.size
    return new_img

def _set_combinations(arr):
    all_subsets = []
    for i in range(1, len(arr)+1):
        all_subsets.extend(combinations(arr, i))
    return all_subsets

# def convert_labels2idx(labels, clusters):
#     for i, c in enumerate(clusters):
#         labels = np.where(labels == c, labels, i+1)
#     return labels

def _standard_deviation(datapoints, mean):
    sigma = np.sqrt(np.mean((datapoints - mean)**2))
    return sigma

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import os
    img = cv2.imread(os.path.join('stored', '2019-06-20-T07-33-48ZCamera-Top-1.jpeg'))
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    lmy = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    l, m, y = lmy[:, :, 0], lmy[:, :, 1], lmy[:, :, 2]
    luv_l, luv_u, luv_v = luv[:, :, 0], luv[:, :, 1], luv[:, :, 2]
    clustered_img, cluster_centers, labels = kmeans_clustering(luv_l, 2)
    plt.imshow(np.array(labels).reshape(g.shape))
    plt.show()
