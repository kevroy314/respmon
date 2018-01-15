import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import time


def show_frequencies(vid_data, fps, bounds=None):
    """Graph the average value of the video as well as the frequency strength"""
    averages = []

    if bounds:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0] - 1):
            averages.append(vid_data[x, :, :, :].sum())

    averages = np.array(averages) - np.array(min(averages))

    charts_x = 1
    charts_y = 2
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=.7)

    plt.subplot(charts_y, charts_x, 1)
    plt.title("Pixel Average")
    plt.xlabel("Time")
    plt.ylabel("Brightness")
    plt.plot(averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / fps)
    fft = abs(scipy.fftpack.fft(averages))
    idx = np.argsort(freqs)

    plt.subplot(charts_y, charts_x, 2)
    plt.title("FFT")
    plt.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    fft = fft[idx]

    freqs = freqs[int(len(freqs) / 2. + 1.):]
    fft = fft[int(len(fft) / 2. + 1.):]
    plt.plot(freqs, abs(fft))

    plt.show()


def reduce_bounding_box(x, y, w, h, maximum_area):
    start_area = w * h
    if start_area <= maximum_area:
        return x, y, w, h
    shrink_proportion = np.sqrt((float(maximum_area) / float(start_area)))
    new_w = w * shrink_proportion
    new_h = h * shrink_proportion
    new_x = x + ((w - new_w) / 2.)
    new_y = y + ((h - new_h) / 2.)
    return int(np.round(new_x)), int(np.round(new_y)), int(np.round(new_w)), int(np.round(new_h))


class Benchmarker:
    def __init__(self):
        self.starts = dict()
        self.ticks = dict()

    def add_tag(self, tag):
        self.ticks[tag] = []

    def tick_start(self, tag):
        self.starts[tag] = time.time()

    def tick_end(self, tag):
        self.ticks[tag].append(time.time() - self.starts[tag])

    def get_report(self):
        return 'Tag, Average Time (seconds), Iterations\r\n' + \
               '\r\n'.join(['{0}, {1}, {2}'.format(tag, rate, iterations) for
                            tag, rate, iterations in zip(self.ticks.keys(),
                                                         [np.mean(l) for l in self.ticks.values()],
                                                         [len(l) for l in self.ticks.values()])])

    def has_tag(self, tag):
        return tag in self.ticks
