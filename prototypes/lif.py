import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import peakutils


def lif(_x, _offset=0, _x_offset=0, _scale=1, _k=1, _t=1.5):
    diff = (np.exp(-_t * _k + _x_offset) * _scale + _offset) - (-np.exp(-_t * _k + _x_offset) * _scale + _offset)
    if _x <= _t:
        val = -np.exp(-_x * _k + _x_offset) * _scale + _offset + diff / 2
    else:
        val = np.exp(-_x * _k + _x_offset) * _scale + _offset - diff / 2
    return np.clip(val + (diff/2), 0, np.inf)


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 10.0, 0.01)
offset = 0
x_offset = 0
scale = 1
k = 1
thresh = 0.5
# s = a0*np.sin(2*np.pi*f0*t)
s = [lif(x, offset, x_offset, scale, k, thresh) for x in t]
l, = plt.plot(t, s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axoffset = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)
axxoffset = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
axscale = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
axk = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
axthresh = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

soffset = Slider(axoffset, 'Y Offset', -10, 10.0, valinit=offset)
sxoffset = Slider(axxoffset, 'X Offset', -10, 10.0, valinit=x_offset)
sscale = Slider(axscale, 'Scale', -10.0, 10.0, valinit=scale)
sk = Slider(axk, 'K', -10.0, 10.0, valinit=k)
sthresh = Slider(axthresh, 'Threshold', 0.0, 10.0, valinit=thresh)


# noinspection PyUnusedLocal
def update(val):
    _offset = soffset.val
    _x_offset = sxoffset.val
    _scale = sscale.val
    _k = sk.val
    _thresh = sthresh.val
    # l.set_ydata([lif(x, _offset, _x_offset, _scale, _k, _thresh) for x in t])
    l.set_ydata([peakutils.gaussian(x, _offset, _x_offset, _scale) for x in t])
    fig.canvas.draw_idle()


soffset.on_changed(update)
sxoffset.on_changed(update)
sscale.on_changed(update)
sk.on_changed(update)
sthresh.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


# noinspection PyUnusedLocal
def reset(event):
    soffset.reset()
    sxoffset.reset()
    sscale.reset()
    sk.reset()
    sthresh.reset()


button.on_clicked(reset)

plt.show()
