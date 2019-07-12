import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def calculate_growth_from_surface_area(surface_area_file, growth_score_file, growth_curve_file):
    surface_area_arr = []
    with open(surface_area_file, 'r') as f:
        for line in sorted(f.readlines()):
            line = line.rstrip()
            if line:
                surface_area = int(line.split(' ')[1])
            else:
                continue
            if surface_area:
                surface_area_arr.append(surface_area)
    # plot_surface_area(surface_area_arr)
    Y = np.array(surface_area_arr)
    # EMA_Y = exponential_moving_average(Y, 0.5)
    SMA_Y = simple_moving_average(Y, 5)
    plot_surface_area(SMA_Y, growth_curve_file)
    X = np.linspace(0, 72, num=SMA_Y.size).reshape(-1, 1)
    model = LinearRegression().fit(X, SMA_Y)
    params = model.coef_
    print(params)
    np.save(growth_score_file, params)

def simple_moving_average(X, window):
    newX = np.empty((X.shape[0]-window,1))
    for idx, x in enumerate(X[window:]):
        newX[idx,:] = np.mean(X[idx:idx+window])
    return newX

def exponential_moving_average(X, alpha):
    X = X.astype(float)
    alpha = np.array(alpha)
    out = np.empty_like(X)
    scaling_factors = np.power(1. - alpha, np.arange(X.size+1))
    np.multiply(X, (alpha * scaling_factors[-2]) / scaling_factors[:-1], out=out)
    np.cumsum(out, out=out)
    out /= scaling_factors[-2::-1]
    return out

# STACKOVERFLOW implementation
def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def plot_surface_area(arr, out_file):
    plt.plot(arr)
    plt.ylabel('Surface area')
    # plt.yticks([])
    plt.xlabel('Time')
    plt.savefig(out_file)
    plt.clf()

if __name__ == "__main__":
    calculate_growth_from_surface_area(os.path.join("..", "..", "results", "surface_area_results.txt"))
