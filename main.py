import logging
from base import RespiratoryMonitor


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)

    data_path = r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\"

    rm = RespiratoryMonitor(capture_target=0, save_calibration_image=True, motion_extraction_method="flow")

'''
    rm = RespiratoryMonitor(capture_target=data_path + "timber.mp4", save_calibration_image=True,
                            motion_extraction_method="flow")
    rm = RespiratoryMonitor(capture_target=data_path + "timber.mp4", save_calibration_image=True,
                            motion_extraction_method="average")
    rm = RespiratoryMonitor(capture_target=data_path + "timber2.mp4", save_calibration_image=True,
                            motion_extraction_method="flow")
    rm = RespiratoryMonitor(capture_target=data_path + "timber2.mp4", save_calibration_image=True,
                            motion_extraction_method="average")
    rm = RespiratoryMonitor(capture_target=data_path + "trooper.mp4", save_calibration_image=True,
                            motion_extraction_method="flow")
    rm = RespiratoryMonitor(capture_target=data_path + "trooper.mp4", save_calibration_image=True,
                            motion_extraction_method="average")
'''
