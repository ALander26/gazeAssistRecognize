import sys
import init_path
import gaze_algorithm as GA
import rcnnModule
import numpy as np
import cv2, os
import imageio
import time
import threading
# from saliency_map import SaliencyMap
# from utils import OpencvIo


_WINSIZE = 5

class CameraObject():
	def __init__(self):
		self.capScene = cv2.VideoCapture(0)
		self.capEye = cv2.VideoCapture(0)
		# self.sceneIm = self.capScene.read()
		self.eyeIm = self.capEye.read()
		self.calibPoints = {}
		self.pupilCenters = {}
		self.LED_centroids = {}
		self.gazeObject = gazeObject

	def update(self, num):
		self.readFrameScene(num);
		# self.readFrameEye(num);

	def readFrameScene(self, num):
		ret, frame = self.capScene.read()
		[frame, points] = self.imProcessingScene(frame)
		self.sceneIm = frame
		self.calibPoints[num] = points
		return frame

	def readFrameEye(self, num):
		ret, frame = self.capEye.read()
		[frame, pupilCenter, LED_centroids] = self.imProcessingEye(frame)

		if pupilCenter == None:
			return self.eyeIm

		self.pupilCenters[num] = pupilCenter
		self.LED_centroids[num] = LED_centroids
		self.eyeIm = frame
		return frame

	def clearObject(self):
		self.capScene.release()
		# self.capEye.release()
		cv2.destroyAllWindows()

	def imProcessingScene(self, frame):
		points = self.getCalibrationPointFromIm(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray, points

	def imProcessingEye(self, im):
		print "Image processing eye"
		gazeObject = self.gazeObject
		grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		# pupilCenter = []
		# LED_centroids = []
		pupilCenter = gazeObject.GetPupilBoundaryPoints(grayIm)
		if pupilCenter == None:
			return im, None, None
		else:		
			LED_centroids = gazeObject.FindLEDCentroids(grayIm)

			im2 = cv2.circle(im, tuple(LED_centroids[0]), 10, (0,0,255), 2)
			im2 = cv2.circle(im2, tuple(LED_centroids[1]), 10, (0,0,255), 2)
			im2 = cv2.circle(im2, tuple(pupilCenter), 10, (0,0,255),2)
			cv2.imshow('frame', im)
			cv2.waitKey(1)

		return grayIm, pupilCenter, LED_centroids

	def getCalibrationPointFromIm(self, im):
		# print "Get Calibration Points from Image"

		# Get Saliency map iffi
	    # sm = SaliencyMap(im)
	    # print sm
	    return im
	    # oi.imshow_array([sm.map])

	def clear(self):
		self.LED_centroids.clear()
		self.calibPoints.clear()
		self.pupilCenters.clear()

	def optimize(self):
		gazeObject = self.gazeObject();
		gazeObject.optimizeGaze(self.calibPoints, self.pupilCenters, self.LED_centroids);
		return

def threadFunc(image, gazeData):
	print "aaaa"
	feature = rcnnObject.getFeatureIm(image, gazeData);

def main():

	tick = 0
	while(True):

		t = time.time()
		tick = tick+1
		cameraObject.update(tick)
		th = threading.Thread(target=threadFunc, args=(cameraObject.sceneIm,))
		th.start()

		cv2.imshow('frame', cameraObject.sceneIm)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# filename = 'dtest.mp4'
	# vid = imageio.get_reader(filename,  'ffmpeg')

	# i=39
	# while(True):
	# 	print i
	# 	image = vid.get_data(i)
	# 	eyeIm = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# 	oriIm = eyeIm.copy();

	# 	# cv2.imshow('frame',eyeIm)
	# 	# cv2.waitKey(0)

	# 	pupilCenter = gazeObject.GetPupilBoundaryPoints(eyeIm)
		
	# 	if pupilCenter==None:
	# 		# Blink state
	# 		print "blink"
	# 		i=i+1
	# 		continue

	# 	# pupilCenter = gazeObject.fitEllipse(x, y)
	# 	LED_centroids = gazeObject.FindLEDCentroids(eyeIm)

	# 	im2 = cv2.circle(oriIm, tuple(LED_centroids[0]), 10, (0,0,255), 2)
	# 	im2 = cv2.circle(im2, tuple(LED_centroids[1]), 10, (0,0,255), 2)
	# 	im2 = cv2.circle(im2, tuple(pupilCenter), 10, (0,0,255),2)
	# 	cv2.imshow('frame',im2)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break
	# 	i=i+1

	# eyeIm = cv2.imread('eye.jpg');
	# oriIm = eyeIm.copy();
	# eyeIm = cv2.cvtColor(eyeIm, cv2.COLOR_BGR2GRAY)

	# tick = 0
	# while(True):

	# 	t = time.time()
	# 	tick = tick+1
	# 	cameraObject.update(tick)

		# print time.time() - t
		# if tick ==_WINSIZE:
		# 	cameraObject.optimize();
		# 	cameraObject.clear()
		# 	tick = 0

		# gazeObject.GetPupilBoundaryPoints(eyeIm)

		# cv2.imshow('frame', cameraObject.sceneIm)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	break


	# x,y = gazeObject.GetPupilBoundaryPoints(eyeIm)

	# pupilCenter = gazeObject.fitEllipse(x, y)
	# print pupilCenter
	# LED_centroids = gazeObject.FindLEDCentroids(eyeIm)

	# pupil_reflection = gazeObject.FindPointOfRefraction()

	# corneal_center = gazeObject.FindCornealCenter(LED_centroids)

	# pupil_center_world = gazeObject.FindPupilCenter(corneal_center, pupil_reflection)

	# print corneal_center
	# print pupil_center_world

	# im2 = cv2.circle(oriIm, tuple(LED_centroids[0]), 10, (0,0,255), 2)
	# im2 = cv2.circle(im2, tuple(LED_centroids[1]), 10, (0,0,255), 2)
	# im2 = cv2.circle(im2, tuple(pupilCenter), 10, (0,0,255),2)
	# cv2.imshow('frame', im2)
	# cv2.waitKey(0)	
	# print centroids
    # Display the resulting frame

	cameraObject.clearObject()


gazeObject = GA.gazeObject();
cameraObject = CameraObject();
rcnnObject = rcnnModule.RcnnObject("vgg_cnn_m_1024", True);

if __name__ == '__main__':
	main();