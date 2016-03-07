import sys
import init_path
import gaze_algorithm as GA
import rcnnModule
import numpy as np
import cv2, os
from utils.timer import Timer
import threading
# from saliency_map import SaliencyMap
# from utils import OpencvIo


_WINSIZE = 5

class CameraObject():
	def __init__(self, gazeObject):
		self.capScene = cv2.VideoCapture(0)
		# self.capEye = cv2.VideoCapture(1)
		self.sceneIm = self.capScene.read()
		# self.eyeIm = self.capEye.read()
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
		# points = self.getCalibrationPointFromIm(frame)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		points = [0,0]
		resized_image = cv2.resize(frame, (500, 375)) 
		gray = resized_image
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
		# gazeObject.optimizeGaze(self.calibPoints, self.pupilCenters, self.LED_centroids);
		return

def threadFunc(image, gazeData):
	# print "aaaa"
	feature = rcnnObject.getFeatureIm(image, gazeData);

def main():
	rcnnModel = rcnnModule.RcnnObject('zf', False);
	gazeObject = GA.gazeObject();
	cam = CameraObject(gazeObject);

	timer = Timer()

	while(True):
		cam.update(0)
		cv2.imshow('frame', cam.sceneIm)
		cv2.waitKey(1)
		# print cam.sceneIm.shape
		timer.tic()		
		[scores, boxes] = rcnnModel.getFeatureIm(cam.sceneIm)
		timer.toc()
		# print scores

if __name__ == '__main__':
	main();