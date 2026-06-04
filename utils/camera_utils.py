import cv2
import config

def FindCamera():
    index = 0
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index
        index += 1
        i -= 1
    return -1

def open_camera(id_camera):
    camera = cv2.VideoCapture(id_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    return camera
