import numpy as np
import itertools
import cv2
from functools import reduce
import matplotlib.pyplot as plt

def dempster_shafer(mass_function, info_sources, frame_of_discernments):
    '''
    Args:
        mass_function: function object to calculate mass
        info_sources: list of information sources, information should be flattened to 1-dim array
        frame_of_discernments: a dictionary of {tuple(subset): int(index)}
    Return:
        numpy array of dim (number_of_subsets_in_frame, info_amount)
    '''
    mass_values = _compute_mass_values(mass_function, info_sources)
    set_dict = _construct_set_dict_from(frame_of_discernments, len(info_sources))
    combined_mass_values = _ds_orthogonal_combinations(mass_values, frame_of_discernments, set_dict)
    return combined_mass_values

def _compute_mass_values(mass_function, info_sources):
    '''
    Args:
        mass_function: function to apply to all information
        info_sources: list of information from different sources
    Returns:
        numpy array of dim (number_of_subsets_in_frame, amount_of_info, info_sources_channel)
    '''
    mass_values_for_all_sources = []
    for source in info_sources:
        mass_values, _ = mass_function(source)
        mass_values_for_all_sources.append(mass_values.squeeze(2))
    mass_values = np.stack(mass_values_for_all_sources).transpose([1, 2, 0])
    return mass_values

def _construct_set_dict_from(frame_set, size):
    subsets = list(frame_set.keys())
    frame_grid = {}
    for combination in itertools.product(subsets, repeat=size):
        intersection = reduce((lambda x, y: set(x) & set(y)), combination)
        frame_grid[combination] = intersection
    return frame_grid

def _ds_orthogonal_combinations(mass_values, frame_of_discernments, set_dict):
    '''
    Based on Dempster's rule of combination
    Args:
        mass_values: numpy array of shape (number_of_subsets_in_frame, amount_of_info, info_sources_channel)
        frame_of_discernments: list of all possible subsets within frame
        set_dict: dictionary of {key: (set1, set2), value: set(intersection_set)}
    Returns:
        combined_mass_values: numpy array of shape (number_of_subsets_in_frame, amount_of_info)
    '''
    subset_frame_size, info_size, info_source_size = mass_values.shape
    combined_mass = np.empty((subset_frame_size, info_size))
    frame_intersections, null_intersections = _build_frame_intersections(frame_of_discernments, set_dict)

    # TODO: refactor this
    mass_conflicts = np.zeros((info_size))
    for set_pair in null_intersections:
        all_m = []
        for s in range(len(set_pair)):
            m = mass_values[frame_of_discernments[set_pair[s]], :, s]
            all_m.append(m)
        mass_conflict = reduce((lambda x, y: np.multiply(x, y)), all_m)
        mass_conflicts += mass_conflict
    normalization_factor = np.ones(info_size) / (np.ones(info_size) - mass_conflicts)

    for idx, set_pair_list in enumerate(frame_intersections):
        mass_supports = np.zeros((info_size))
        for set_pair in set_pair_list:
            all_m = []
            for s in range(len(set_pair)):
                m = mass_values[frame_of_discernments[set_pair[s]], :, s]
                all_m.append(m)
            mass_support = reduce((lambda x, y: np.multiply(x, y)), all_m)
            mass_supports += mass_support
        combined_mass[idx, :] = np.multiply(normalization_factor, mass_supports)

    return combined_mass

def _build_frame_intersections(frame_of_discernments, set_dict):
    frame_intersections = []
    null_intersections = []
    for idx, frame in enumerate(frame_of_discernments.keys()):
        intersections = []
        for key, val in set_dict.items():
            if not val:
                null_intersections.append(key)
            elif set(frame) == val:
                intersections.append(key)
        frame_intersections.append(intersections)
    return frame_intersections, null_intersections

def _apply_decisional_procedure(mass_values):
    # Torch.argmax
    # if
    return

def display_img(img):
    # output = np.array([0, 123, 123])
    # unique, counts = np.unique(output[img], return_counts=True)
    # print(unique, counts)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    # frame_set = [(1,), (2,), (1, 2)]
    # set_grid = _construct_set_dict_from(frame_set, 3)
    # intersect, null = build_frame_intersections(frame_set, set_grid)
    # print(set_grid)
    # print(len(set_grid))
    # print(intersect)
    # print(null)
    import cv2
    import matplotlib.pyplot as plt
    import os
    from kmeans import kmeans_mass_function
    img = cv2.imread(os.path.join('stored', '2019-06-20-T07-33-48ZCamera-Top-1.jpeg'))
    rgb_r, rgb_g, rgb_b = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lmy = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

    hsv_h, hsv_s, hsv_v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lmy_l, lmy_m, lmy_y = lmy[:, :, 0], lmy[:, :, 1], lmy[:, :, 2]
    xyz_x, xyz_y, xyz_z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    ycrcb_y, ycrcb_cr, ycrcb_cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    hls_h, hls_l, hls_s = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
    luv_l, luv_u, luv_v = luv[:, :, 0], luv[:, :, 1], luv[:, :, 2]
    color_dict = {'hsv_h':hsv_h, 'hsv_s':hsv_s, 'hsv_v':hsv_v,
             'rgb_r':rgb_r, 'rgb_g':rgb_g, 'rgb_b':rgb_b,
             'xyz_x':xyz_x, 'xyz_y':xyz_y, 'xyz_z':xyz_z,
             'ycrcb_y':ycrcb_y, 'ycrcb_cr':ycrcb_cr, 'ycrcb_cb':ycrcb_cb,
             'hls_h':hls_h, 'hls_l':hls_l, 'hls_s':hls_s,
             'luv_l':luv_l, 'luv_u':luv_u, 'luv_v':luv_v,
             'lmy_l':lmy_l, 'lmy_m':lmy_m, 'lmy_y':lmy_y,}
    frame_set = {(1,):0, (2,):1, (1, 2):2}
    combined_mass = dempster_shafer(kmeans_mass_function, [luv_l, lmy_y], frame_set)
    combined_mass_argmax = np.argmax(combined_mass, axis=0)
    # print(combined_mass_argmax)
    h, w = hsv_h.shape
    output_img = combined_mass_argmax.reshape((h-2, w-2))
    np.save("dempster_img.npy", output_img)
    display_img(np.load("dempster_img.npy"))
