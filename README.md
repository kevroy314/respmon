# Respmon

This repository contains Python code for real-time respiration monitoring using a webcam. The code began based on [an implementation](https://github.com/brycedrennan/eulerian-magnification) of [Eulerian Video Magnification](https://people.csail.mit.edu/mrub/papers/vidmag.pdf) in Python, but that technique prooved to be too computationally expensive and is now used as a calibration step. I designed and built this so that animal shelters could have a low-cost real-time monitoring system for critical animals in a lethargic/comatose state. Specifically, it was built with [Austin Pets Alive](https://www.austinpetsalive.org/) in mind.

## Setup

It's recommended you run this in an [Anaconda](https://www.anaconda.com/download/) virtual environment. This can be configured by running:

    conda create -n respmon python=3.5 matplotlib numpy scipy pyqtgraph tqdm pywavelets
    activate respmon
    pip install peakutils
    conda install --channel https://conda.anaconda.org/menpo opencv3
    git clone https://github.com/kevroy314/respmon .
    python main.py

This will run the main file, which you can modify to determine the video source (0 is likely your webcam).

## Algorithm Description

The core algorithm is in base.py and is described in this section. The RespiratoryMonitor object contains a state machine which is initalized using the run() function. The state machine operates as follows:

![State Diagram](https://github.com/kevroy314/respmon/raw/master/images/state.png)
  
### Calibration

Calibration is performed by acquiring some frames and performing eulerian magnification in order to isolate a rectangular Region of Interest (ROI) in which the right frequency properties are present for a respiratory signal to be measured. It is computationally expensive compared to the rest of the algorithm and is only run at the beginning and if the signal is no longer viable.

Eulerian magnification is performed by first constructing a [Laplacian-Gaussian image pyramid](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html) for each frame in the video, and then performing a [Fourier Transform](https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html) along the temporal axis for each pixel at each scale. 

![Laplacian Pyramid Image](https://github.com/kevroy314/respmon/raw/master/images/pyramid2.png)
![Filtered Pyramid Image](https://github.com/kevroy314/respmon/raw/master/images/pyramid.png)

The resulting frequency pyramid is then collapsed and averaged along the temporal axis to get a single image which acts as a "heatmap" for the locations in the image where a particular frequency is common. This image is then thresholded and the [largest contour](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features) is isolated for later measurement. This largest contour's bounding box is the ROI.

![Calibration Steps Image](https://github.com/kevroy314/respmon/raw/master/images/calibration0.png)

### Motion Measurement

Motion Measurement is performed in one of two ways depending on the configuration, either a simple average of the ROI pixel values is used or optical flow is performed. The averaging method is computationally inexpensive, but prone to errors depending on the textures/colors present in the frames. It does tend to produce a smoother signal when it works. It also has a scale which depends on the image contents, making it harder to detect errors consistently.

For [optical flow](https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html), a set of [feature points](https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html) are identified. These feature points are then tracked across frames and the difference between consecutive points becomes a raw motion signal. The [first eigenvector](http://sebastianraschka.com/Articles/2014_pca_step_by_step.html) of this raw motion signal is extracted at each time step and used to transform the 2D motion vector average into a 1D signal which can be used as a measure of the motion along the primary axis of motion. This method is more computationally expensive but produces a signal which represents the net motion in units of pixels, making it an easier signal within which to detect errors.

[![Motion Estimation](https://github.com/kevroy314/respmon/raw/master/images/motion.gif)](https://www.youtube.com/watch?v=T8MH772fuOo)

### Frequency/Peak Detection

Because the signals are low frequency, it's more reliable to [measure frequency via peak-to-peak intervals](https://gist.github.com/endolith/255291) than other fourier-based methods. To reliably detect peaks, first, the signal is [lowpass filtered](https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units) at half of the maximum frequency used in the localizer (creating a far smoother signal). The filtered signal then passes through a [peak detection](https://blog.ytotech.com/2015/11/01/findpeaks-in-python/) algorithm whose results are filtered based on a minimum width constraint. The resulting candidate peaks are fit with a gaussian curve (which can fail, resulting in the removal of that candidate peak). The curvature parameter of the gaussian fit must be below a certain threshold for the peak to pass and be sent on as an actual, measured peak (this avoiding mini-peaks and false positives on rising edges). 

![Peak Detection](https://github.com/kevroy314/respmon/raw/master/images/peaks.png)

### Final Signal

The final output signal is one over the average interval within the measurement buffer. This result is presented in Beats Per Minute (BPM) and plotted for convenience.

[![Motion Estimation](https://github.com/kevroy314/respmon/raw/master/images/measuring.gif)](https://www.youtube.com/watch?v=lylg_yagLpE)
