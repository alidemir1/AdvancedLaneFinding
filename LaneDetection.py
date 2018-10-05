import cv2 as cv
import numpy as np
import time
import LaneDetector as LD

cap = cv.VideoCapture('project_video.mp4')
width = cap.get(3)
height = cap.get(4)
print(width, height)

#Video Writer
# Define the codec and create VideoWriter object
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output.avi',fourcc, 25.0, (1280,720))






numOfBoxEachLine = 6
frameNum = 0
lastFrameNumL = -10000
lastFrameNumR = -10000
lastLeftLaneIndiceAtBottom = 1
lastRightLaneIndiceAtBottom = width - 1
lastLeftLaneIndices = np.zeros((numOfBoxEachLine))
lastRightLaneIndices = np.zeros((numOfBoxEachLine))
lastLeftLaneIndicesMean = 0.0
lastRightLaneIndicesMean = 0.0
roadWidth = 0.0
lastFrameNumPoly = -10000
roadCurvlastTen = np.zeros(20)

while(cap.isOpened()):
	tic = time.time()
	#reads video frame

	ret, frame = cap.read()
	frameNum += 1
	roadCurvlastTen, lastLeftLaneIndiceAtBottom, lastRightLaneIndiceAtBottom, lastFrameNumL, lastFrameNumR, roadWidth, lastLeftLaneIndicesMean, lastRightLaneIndicesMean, lastLeftLaneIndices, lastRightLaneIndices = LD.laneDetector(frame, frameNum, roadCurvlastTen, lastLeftLaneIndiceAtBottom, lastRightLaneIndiceAtBottom, lastFrameNumL, lastFrameNumR, roadWidth, lastLeftLaneIndicesMean, lastRightLaneIndicesMean, lastLeftLaneIndices, lastRightLaneIndices)
	cv.imshow("Lane Detector", frame)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

	print("FPS: ", 1.0 / (time.time() - tic))

cap.release()
cv.destroyAllWindows()



