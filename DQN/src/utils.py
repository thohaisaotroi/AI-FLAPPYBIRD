import cv2
import numpy as np

def xuLyAnh(anhGoc, chieuRong, chieuCao):
    anhXam = cv2.cvtColor(cv2.resize(anhGoc, (chieuRong, chieuCao)), cv2.COLOR_BGR2GRAY)
    _, anhNhiPhan = cv2.threshold(anhXam, 1, 255, cv2.THRESH_BINARY)
    return anhNhiPhan[None, :, :].astype(np.float32)
