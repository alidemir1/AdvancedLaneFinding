import cv2 as cv
import numpy as np
import os

def calibrationParameters(path, cshape):
    """Compute calibration parameters from a set of calibration images.
    Params:
      path: Directory of calibration images.
      cshape: Shape of grid used in the latter.
    Return:
      mtx, dist
    """
    # Object / image points collections.
    objpoints = []
    imgpoints = []

    # Calibration points from images.
    filenames = os.listdir(path)
    for fname in filenames:
        img = cv.imread(path + fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Theoretical Grid.
        objp = np.zeros((cshape[0] * cshape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:cshape[0], 0:cshape[1]].T.reshape(-1, 2)
        # Corners in the image.
        ret, corners = cv.findChessboardCorners(gray, cshape, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print('Warning! Not chessboard found in image', fname)
    # Calibration from image points.
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       img.shape[0:2],
                                                       None, None)
    return mtx, dist