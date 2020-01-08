import cv2 as cv
import numpy as np


ipmMargin = 90 #110 50 90

heightOfRoi = 200 #120 160 260
numOfBoxEachLine = 6



#box is used to find line features with going through to top of the image, with the help of the bottom lane feature
box = np.ones((int(heightOfRoi/numOfBoxEachLine), 10)) 



leftLaneIndices = np.zeros((numOfBoxEachLine))
rightLaneIndices = np.zeros((numOfBoxEachLine))

def laneDetector(frame, frameNum, roadCurvlastTen, lastLeftLaneIndiceAtBottom, lastRightLaneIndiceAtBottom, lastFrameNumL, lastFrameNumR, roadWidth, lastLeftLaneIndicesMean, lastRightLaneIndicesMean, lastLeftLaneIndices, lastRightLaneIndices, mtx, dist):
	midPointofLane = 0.0
	polyVariance = 0.0

	roi = frame[frame.shape[0] - heightOfRoi:int(frame.shape[0]), 0:int(frame.shape[1])] #120 160
	cv.imshow('ROI', roi)


	#kernel is bottom band to find line features at the bottom of the image
	kernel = np.ones((40,int(frame.shape[1]))) 
	#performs inverse perspective mapping (ipm): in the order of left top, right top,
	#left bottom, right bottom
	src = np.array([[0,0],[roi.shape[1],0],[0,roi.shape[0]],[roi.shape[1],roi.shape[0]]],np.float32)
	dst = np.array([[0,0],[roi.shape[1],0],[roi.shape[1]/2 - ipmMargin,roi.shape[0]],[roi.shape[1]/2 + ipmMargin,roi.shape[0]]],np.float32) 
	M = cv.getPerspectiveTransform(src, dst)

	ipmed = cv.warpPerspective(roi, M, (roi.shape[1], roi.shape[0]))
	
	ipmed[:,:int(roi.shape[1]/2 - (ipmMargin + 100))] = 0
	ipmed[:, int(roi.shape[1]/2 + (ipmMargin + 100)):] = 0
	#cv.imshow('ipmed', ipmed)

	#show ipm on frame
	frame[125:ipmed.shape[0]+125, 0:2*(ipmMargin+100)] = ipmed[:,int(roi.shape[1]/2 - (ipmMargin + 100)):int(roi.shape[1]/2 + (ipmMargin + 100))]
	


	#convertImageintoGrayScale
	grayIPM = cv.cvtColor(ipmed, cv.COLOR_BGR2GRAY)	

	CgrayIPM = cv.Canny(grayIPM, 100,200)
	#cv.imshow("Canny", CgrayIPM)
	
	#Calculates sobel gradients
	sobelx = cv.Sobel(CgrayIPM,cv.CV_64F,1,0,ksize=5)
	#cv.imshow("sobelx IPM", sobelx)
	

	morphkernel = np.ones((13,13),np.uint8) # 13,13   4,4
	sobelx = cv.morphologyEx(sobelx, cv.MORPH_CLOSE, morphkernel)
	#cv.imshow('Morph', sobelx)

	#to remove the IPM borders
	sobelx[:,:int(sobelx.shape[1]/2 - (ipmMargin+95))] = 0
	sobelx[:,int(sobelx.shape[1]/2 + (ipmMargin+95)):] = 0

	#remove car
	sobelx[sobelx.shape[0] - 25:sobelx.shape[0],:] = 0

	#cv.imshow('MorphBordersRemoved', sobelx)	
	


	kernelOutput = np.sum(sobelx[sobelx.shape[0] - kernel.shape[0]:, :]*kernel, axis = 0)
	#print(kernelOutput.shape)

	LeftKernelOutput = kernelOutput[:int(kernelOutput.shape[0]/2)]
	RightKernelOutput = kernelOutput[int(kernelOutput.shape[0]/2):]
	#print(LeftKernelOutput)

	#maximum number in every row (candidate lane features) of the kerneloutput matrice
	leftLaneIndiceAtBottom = np.argmax(LeftKernelOutput) #* 64 
	rightLaneIndiceAtBottom = np.argmax(RightKernelOutput) + int(sobelx.shape[1]/2) #* 64 + int(sobelx.shape[1]/2)

	if (lastLeftLaneIndiceAtBottom >= 10 and lastRightLaneIndiceAtBottom <= frame.shape[1] - 10):
		leftLaneIndiceAtBottom = np.argmax(LeftKernelOutput[lastLeftLaneIndiceAtBottom - 5:lastLeftLaneIndiceAtBottom + 5]) + (lastLeftLaneIndiceAtBottom - 5)
		rightLaneIndiceAtBottom = np.argmax(RightKernelOutput[(lastRightLaneIndiceAtBottom - int(sobelx.shape[1]/2)) - 5:(lastRightLaneIndiceAtBottom  - int(sobelx.shape[1]/2)) + 5]) + ((lastRightLaneIndiceAtBottom - int(sobelx.shape[1]/2)) - 5) + int(sobelx.shape[1]/2)


	if polyVariance >= 0.89:     #frameNum - lastFrameNumPoly <= 300:
		if np.sum(sobelx[sobelx.shape[0] - kernel.shape[0]:, int(lastLeftLaneIndiceAtBottom - 5): int(lastLeftLaneIndiceAtBottom + 5)]) > box.shape[0]:
			LeftKernelOutput = np.sum(sobelx[sobelx.shape[0] - kernel.shape[0]:, int(lastLeftLaneIndiceAtBottom - 5): int(lastLeftLaneIndiceAtBottom + 5)], axis = 0)
			leftLaneIndiceAtBottom = np.argmax(LeftKernelOutput) + int(lastLeftLaneIndiceAtBottom - 5)

		if np.sum(sobelx[sobelx.shape[0] - kernel.shape[0]:, int(lastRightLaneIndiceAtBottom - 5): int(lastRightLaneIndiceAtBottom + 5)]) > box.shape[0]: 	
			RightKernelOutput = np.sum(sobelx[sobelx.shape[0] - kernel.shape[0]:, int(lastRightLaneIndiceAtBottom - 5): int(lastRightLaneIndiceAtBottom + 5)], axis = 0)
			rightLaneIndiceAtBottom = np.argmax(RightKernelOutput) + int(lastRightLaneIndiceAtBottom - 5)

	

	#in order to prevent dashed lines gaps from being features
	if (leftLaneIndiceAtBottom <= roi.shape[1]/2 - ipmMargin or leftLaneIndiceAtBottom >= roi.shape[1]/2 or polyVariance < 0.89) and frameNum - lastFrameNumL <= 100:
		leftLaneIndiceAtBottom = lastLeftLaneIndiceAtBottom

	if (rightLaneIndiceAtBottom >= roi.shape[1]/2 + ipmMargin or rightLaneIndiceAtBottom <= roi.shape[1]/2 or polyVariance < 0.89) and frameNum - lastFrameNumR <= 100:
		rightLaneIndiceAtBottom = lastRightLaneIndiceAtBottom


	print(str(leftLaneIndiceAtBottom) + ", " + str(rightLaneIndiceAtBottom))


	leftLaneIndices[0] = leftLaneIndiceAtBottom
	rightLaneIndices[0] = rightLaneIndiceAtBottom

	
	if leftLaneIndices[0] != 0:
		for i in range(1,numOfBoxEachLine):
			#threshold of being left lane feature
			if np.sum(sobelx[sobelx.shape[0] - (i+1)*box.shape[0]:sobelx.shape[0] - i*box.shape[0], int(leftLaneIndices[i-1]) - int(box.shape[1]/2):int(leftLaneIndices[i-1]) + int(box.shape[1]/2)]) > box.shape[0]: 
				leftLaneIndices[i] = np.argmax(np.sum(box*sobelx[sobelx.shape[0] - (i+1)*box.shape[0]:sobelx.shape[0] - i*box.shape[0], int(leftLaneIndices[i-1]) - int(box.shape[1]/2):int(leftLaneIndices[i-1]) + int(box.shape[1]/2)], axis = 0)) - int(box.shape[1]/2) + leftLaneIndices[i-1]
			else:
				leftLaneIndices[i] = 0	
			#threshold of being right lane feature
			if np.sum(sobelx[sobelx.shape[0] - (i+1)*box.shape[0]:sobelx.shape[0] - i*box.shape[0], int(rightLaneIndices[i-1]) - int(box.shape[1]/2):int(rightLaneIndices[i-1]) + int(box.shape[1]/2)]) > box.shape[0]:
				rightLaneIndices[i] = np.argmax(np.sum(box*sobelx[sobelx.shape[0] - (i+1)*box.shape[0]:sobelx.shape[0] - i*box.shape[0], int(rightLaneIndices[i-1]) - int(box.shape[1]/2):int(rightLaneIndices[i-1]) + int(box.shape[1]/2)], axis = 0)) - int(box.shape[1]/2) + rightLaneIndices[i-1]
			else:
				rightLaneIndices[i] = 0	


	#Polynomial Fitting
	Ypoints = np.linspace(roi.shape[0], 0, numOfBoxEachLine, True)

	YL = Ypoints[np.where(leftLaneIndices != 0)]
	YR = Ypoints[np.where(rightLaneIndices != 0)]
	#print(Ypoints.shape)

	if len(leftLaneIndices[np.where(leftLaneIndices != 0)]) != 0 and len(rightLaneIndices[np.where(rightLaneIndices != 0)]) != 0:
		leftpoly = np.polyfit(YL, leftLaneIndices[np.where(leftLaneIndices != 0)], 2)
		rightpoly = np.polyfit(YR, rightLaneIndices[np.where(rightLaneIndices != 0)], 2)
		
		print("rightPoly:", rightLaneIndices[np.where(rightLaneIndices != 0)])

		L = np.poly1d(leftpoly)
		R = np.poly1d(rightpoly)


		polyDifference = np.zeros((ipmed.shape[0]))
		q = np.arange(0, ipmed.shape[0])
		polyDifference = R(q) - L(q)
		polyVariance = np.amin(polyDifference)/np.amax(polyDifference)

		print("Poly variance: ",polyVariance) 



		curvatureFlag = 0 #this flag use to specify which line should be used to calculate curvature
		for k in range(0, ipmed.shape[0]):
			#if both left and right lane indices lengths more than 2
			if len(YL) > 2 and len(YR) > 2:
				ipmed[k, int(L(k)):int(L(k))+6] = (0,0,255)
				ipmed[k, int(R(k)):int(R(k))+6] = (0,0,255)
				ipmed[k, int(L(k))+6:int(R(k)),1] = 200
				#mid point of the lane
				midPointofLane = (R(0) - L(0))/2 + L(0)
			
			#if left lane indices are known poorly but right lane indices known well
			elif len(YL) <= 2 and len(YR) > 2 and roadWidth > 80:
				#ipmed[k, int(L(k)):int(L(k))+6] = (0,0,255)
				ipmed[k, int(R(k)):int(R(k))+6] = (0,0,255)
				ipmed[k, int(R(k)) - int(roadWidth):int(R(k)),1] = 200	
				midPointofLane = R(0) - roadWidth/2
				curvatureFlag = 1
				
			#if right lane indices known poorly but left lan indices known well
			elif len(YL) > 2 and len(YR) <= 2 and roadWidth > 80:
				ipmed[k, int(L(k)):int(L(k))+6] = (0,0,255)
				#ipmed[k, int(R(k)):int(R(k))+6] = (0,0,255)
				ipmed[k, int(L(k)):int(L(k))+6+int(roadWidth),1] = 200
				midPointofLane = L(0) + roadWidth/2
				curvatureFlag = 2
		

		#Road Curvature
		YgroundTruth = 30/ frame.shape[0] #meters per pixel
		XgroundTruth = 6.76 / frame.shape[1]
		if len(YL) > 2:
			leftpolyGT = np.polyfit(YgroundTruth*YL, XgroundTruth*leftLaneIndices[np.where(leftLaneIndices != 0)], 2)
			#radius of curvature in meters
			leftLaneCurvature = ((1 + (2*leftpolyGT[0]*np.amax(YL)*YgroundTruth + leftpolyGT[1])**2)**1.5)/np.absolute(2*leftpolyGT[0])
		else:
			leftLaneCurvature = 10**30

		if len(YR) > 2:
			rightpolyGT = np.polyfit(YgroundTruth*YR, XgroundTruth*rightLaneIndices[np.where(rightLaneIndices != 0)], 2)
			rightLaneCurvature = ((1 + (2*rightpolyGT[0]*np.amax(YR)*YgroundTruth + rightpolyGT[1])**2)**1.5)/np.absolute(2*rightpolyGT[0])
		else:
			rightLaneCurvature = 10**30

		roadCurvlastTen[frameNum % 20] = np.minimum(leftLaneCurvature, rightLaneCurvature)
		if curvatureFlag == 1:
			roadCurvlastTen[frameNum % 20] = rightLaneCurvature
		elif curvatureFlag == 2:
			roadCurvlastTen[frameNum % 20] = leftLaneCurvature

		#moving average for road curvature
		roadCurvature = np.mean(roadCurvlastTen[np.where(roadCurvlastTen != 0)])

		locationOfVehicleOnLane = (frame.shape[1]/2 - midPointofLane)*XgroundTruth #in meters
		side = ""
		if locationOfVehicleOnLane > 0:
			side = "left of the lane"
		elif locationOfVehicleOnLane < 0:
			side = "right of the lane"


		#bluen the backgroung of the text
		frame[0:125, :, :] = (139,0,0)
		#put text to the frame
		font = cv.FONT_HERSHEY_SIMPLEX
		cv.putText(frame, "Radius of Curvature: " + str(np.around(roadCurvature,2)) + "m", (400, 30), font, 0.8, (0,255,255), 1)
		cv.putText(frame, "Vehicle is at " + str(np.absolute(np.around(locationOfVehicleOnLane, 2))) + "m " + side + " center", (400, 60), font, 0.8, (0,255,255), 1)
		cv.putText(frame, "Inverse Perspective Mapping (IPM)", (5, 110), font, 0.8, (0,255,255), 1)
		cv.putText(frame, "Filtered IPM", (1000, 110), font, 0.8, (0,255,255), 1)
	
	#cv.imshow("Lines", ipmed)

	#show sobelx on the last frame with lane found
	frame[125:ipmed.shape[0]+125, frame.shape[1]-2*(ipmMargin+100):,0] = sobelx[:,int(roi.shape[1]/2 - (ipmMargin + 100)):int(roi.shape[1]/2 + (ipmMargin + 100))]
	frame[125:ipmed.shape[0]+125, frame.shape[1]-2*(ipmMargin+100):,1] = sobelx[:,int(roi.shape[1]/2 - (ipmMargin + 100)):int(roi.shape[1]/2 + (ipmMargin + 100))]
	frame[125:ipmed.shape[0]+125, frame.shape[1]-2*(ipmMargin+100):,2] = sobelx[:,int(roi.shape[1]/2 - (ipmMargin + 100)):int(roi.shape[1]/2 + (ipmMargin + 100))]



	#Visualize lane on original image
	unipmed = cv.warpPerspective(ipmed, M, (roi.shape[1], roi.shape[0]), flags = cv.WARP_INVERSE_MAP + cv.INTER_NEAREST, borderValue = 0)
	
	unipmedComplement = np.zeros((frame.shape))
	unipmedComplement[frame.shape[0] - roi.shape[0]:,:,:] = unipmed

	nonzeros = unipmedComplement != 0
	frame[nonzeros] = unipmedComplement[nonzeros]



	indOfGreenAreaIpmed = (ipmed[:,:,1] == 200) * 1
	indOfGreenArea = cv.warpPerspective(indOfGreenAreaIpmed, M, (roi.shape[1], roi.shape[0]), flags = cv.WARP_INVERSE_MAP + cv.INTER_NEAREST, borderValue = 0)

	print("indOfGreenShape: ", indOfGreenArea.shape)
	for i in range(0, heightOfRoi - 10, 30):
		if i >= roi.shape[0]:
			i = roi.shape[0] - 1
		greenAtRow = indOfGreenArea[i,:]
		firstGreenLocationAtRow = np.argmax(greenAtRow == 1)#[0]
		shift = frame.shape[0] - heightOfRoi
		print("firstGreenLocationAtRow: ", firstGreenLocationAtRow)
		cv.rectangle(frame, (firstGreenLocationAtRow , i - int(30 * i/15) + shift), (firstGreenLocationAtRow + np.sum(greenAtRow), i + shift), (0,200,0), 1)

	#cv.imshow("Frame", frame)
	#writes frame to a video file
	#out.write(frame)
	
	if leftLaneIndiceAtBottom > roi.shape[1]/2 - ipmMargin and leftLaneIndiceAtBottom < roi.shape[1]/2 and polyVariance > 0.89:
		lastLeftLaneIndiceAtBottom = leftLaneIndiceAtBottom
		lastFrameNumL = frameNum

	if rightLaneIndiceAtBottom < roi.shape[1]/2 + ipmMargin and rightLaneIndiceAtBottom > roi.shape[1]/2 and polyVariance > 0.89:
		lastRightLaneIndiceAtBottom = rightLaneIndiceAtBottom
		lastFrameNumR = frameNum

	#if two polynomials are almost parallel store the roadWidth
	if polyVariance > 0.98:
		roadWidth = lastRightLaneIndiceAtBottom - leftLaneIndiceAtBottom
		print("roadWidth:", roadWidth)

	lastLeftLaneIndicesMean = np.mean(leftLaneIndices)
	lastRightLaneIndicesMean = np.mean(rightLaneIndices)
	lastLeftLaneIndices = leftLaneIndices
	lastRightLaneIndices = rightLaneIndices

	return roadCurvlastTen, lastLeftLaneIndiceAtBottom, lastRightLaneIndiceAtBottom, lastFrameNumL, lastFrameNumR, roadWidth, lastLeftLaneIndicesMean, lastRightLaneIndicesMean, lastLeftLaneIndices, lastRightLaneIndices