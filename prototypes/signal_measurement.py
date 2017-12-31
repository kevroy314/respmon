import numpy as np
import matplotlib.pyplot as plt
import peakutils
from scipy.signal import butter, filtfilt

measure_buffer_length = 128  # Hyperparameter (from main.py)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def find_peaks(filename, ax, fs):
    data_dir = r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\"
    t, data = np.transpose(np.load(data_dir + filename))
    ax.plot(t, data)
    width = 5

    order = 3
    cutoff = 1.0
    data = butter_lowpass_filter(data, cutoff, fs, order)

    indices = peakutils.indexes(data, min_dist=width)
    peak_ts = np.take(t, indices)
    peak_ds = np.take(data, indices)
    ax.plot(t, data)
    ax.scatter(peak_ts, peak_ds, c='r', alpha=0.5)

    final_idxs = []
    post_gaussian = []
    for idx in indices:
        w = width
        if idx - width < 0:
            w = idx
        if idx + w > len(t):
            w = len(t) - idx
        ti = t[idx - w:idx + w]
        datai = data[idx - w:idx + w]
        # ax.plot(ti, datai)
        try:
            params = peakutils.gaussian_fit(ti, datai, center_only=False)
            y = [peakutils.gaussian(x, *params) for x in ti]
            # ax.plot(ti, y)
            ssr = np.sum(np.power(np.subtract(y, datai), 2.0))
            sst = np.sum(np.power(np.subtract(y, datai), 2.0))
            r2 = 1 - (ssr / sst)
            post_gaussian.append(idx)
            if params[2] < 10.0:
                final_idxs.append(idx)
                print('idx={0}, r2={1}, params={2}'.format(idx, r2, params))
        except RuntimeError:
            pass
    peak_ts = np.take(t, post_gaussian)
    peak_ds = np.take(data, post_gaussian)
    ax.scatter(peak_ts, peak_ds, c='g', alpha=0.75)
    peak_ts = np.take(t, final_idxs)
    peak_ds = np.take(data, final_idxs)
    ax.scatter(peak_ts, peak_ds, c='b', alpha=1.0)
    for i, txt in enumerate(final_idxs):
        ax.annotate(str(txt), (peak_ts[i], peak_ds[i]))


def find_peaks_simplified(data, t, width=5, gaussian_cutoff=10.0):
    indices = peakutils.indexes(data, min_dist=width)

    final_idxs = []
    fits = []
    for idx in indices:
        w = width
        if idx - width < 0:
            w = idx
        if idx + w > len(t):
            w = len(t) - idx
        ti = np.array(t)[idx - w:idx + w]
        datai = np.array(data)[idx - w:idx + w]
        try:
            params = peakutils.gaussian_fit(ti, datai, center_only=False)
            y = [peakutils.gaussian(x, *params) for x in ti]
            # ax.plot(ti, y)
            ssr = np.sum(np.power(np.subtract(y, datai), 2.0))
            sst = np.sum(np.power(np.subtract(y, datai), 2.0))
            r2 = 1 - (ssr / sst)
            fits.append(r2)
            if params[2] < gaussian_cutoff:
                final_idxs.append(idx)
        except RuntimeError:
            pass
    return final_idxs, fits


if __name__ == "__main__":
    files = ['timber.mp4.npy', 'timber2.mp4.npy']
    fss = [5.01, 7.68]
    fig, axes = plt.subplots(1, len(files))
    fig.set_size_inches(18.5, 10.5)
    [find_peaks(f, a, s) for f, a, s in zip(files, axes, fss)]
    plt.show()
