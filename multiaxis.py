import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.mkQApp()

pw = pg.PlotWidget()
pw.show()
pw.setWindowTitle('pyqtgraph example: MultiplePlotAxes')
p1 = pw.plotItem
p1.setLabels(left='axis 1')

## create a new ViewBox, link the right axis to its coordinate system
p2 = pg.ViewBox()
p1.showAxis('right')
p1.scene().addItem(p2)
p1.getAxis('right').linkToView(p2)
p2.setXLink(p1)
p1.getAxis('right').setLabel('axis2', color='#0000ff')


def updateViews():
    global p1, p2, p3
    p2.setGeometry(p1.vb.sceneBoundingRect())
    p2.linkedViewChanged(p1.vb, p2.XAxis)


updateViews()
p1.vb.sigResized.connect(updateViews)

p1.plot([1, 2, 4, 8, 16, 32])
p2.addItem(pg.PlotCurveItem([10, 20, 40, 80, 40, 20], pen='b'))

if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
