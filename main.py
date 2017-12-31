import logging
from base import RespiratoryMonitor


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(message)s", level=logging.INFO)

    data_path = r"C:\Users\kevin\Desktop\Active Projects\Video Magnification Videos\\"
    # rm = RespiratoryMonitor(capture_target=data_path + "timber.mp4", save_calibration_image=True)
    # rm = RespiratoryMonitor(capture_target=data_path + "timber2.mp4", save_calibration_image=False)
    # rm = RespiratoryMonitor(capture_target=0, save_calibration_image=False, show_error_signal=True)
    rm = RespiratoryMonitor(capture_target=data_path + "trooper.mp4", save_calibration_image=True, save_all_data=True)
