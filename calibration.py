import sys
import init_path
import gaze_algorithm as GA
import numpy as np
import cv2, os
import wx
import time
import csv

class CameraObject():
	def __init__(self, gazeObject):

		self.capScene = cv2.VideoCapture(0)
		if not self.capScene.isOpened():
			raise NameError("Scene Camera don`t connected")
			return
		# self.capEye = cv2.VideoCapture(1)
		# if not self.capEye.isOpened():
		# 	raise NameError("Eye Camera don`t connected")
		# 	return

		ret, self.sceneIm = self.capScene.read()
		self.sceneThresh = None
		# ret, self.eyeIm = self.capEye.read()
		self.calibPoints = []
		self.calibPointsLabel = []
		self.pupilCenters = []
		self.LED_centroids = []
		self.gazeObject = gazeObject

	def update(self, drawLabel):
		points = self.readFrameScene();
		# pupilCenter, LED_centroid = self.readFrameEye();

		# if pupilCenter is None or points is None:
		# 	return
		if points is None:
			return
		self.calibPoints.append(points)
		self.calibPointsLabel.append(drawLabel)
		# self.pupilCenters.append(pupilCenter)
		# self.LED_centroids.append(LED_centroid)
		return

	def readFrameScene(self):
		scene_threshold = 232
		ret, frame = self.capScene.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		ret, thresh = cv2.threshold(gray, scene_threshold, 255, 0)
		self.sceneIm = frame
		self.sceneThresh = thresh

		points = self.imProcessingScene(thresh)

		return points

	def imProcessingScene(self, threshIm):

		# out = self.sceneIm.copy()
		circles = cv2.HoughCircles(threshIm, cv2.cv.CV_HOUGH_GRADIENT, 1.1, 2000,
			param1=30,
			param2=5,
			minRadius=5,
			maxRadius=8)

		# print circles[0, 0]
		if circles is None:
			return None
		else:
			return circles[0, 0]

	def readFrameEye(self, num):
		ret, frame = self.capEye.read()

		if ret == False:
			return None

		[frame, pupilCenter, LED_centroids] = self.imProcessingEye(frame)

		if pupilCenter == None | len(LED_centroids) < 2:
			return self.eyeIm

		self.pupilCenters[num] = pupilCenter
		self.LED_centroids[num] = LED_centroids


		# frame = self.imProcessingEye(frame)

		self.eyeIm = frame
		return frame

	def clearObject(self):
		self.capScene.release()
		self.capEye.release()
		cv2.destroyAllWindows()

	def imProcessingEye(self, im):
		gazeObject = self.gazeObject
		grayIm = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		pupilCenter = gazeObject.GetPupilBoundaryPoints(grayIm)
		if pupilCenter == None:
			return im, None, None
		else:		
			LED_centroids = gazeObject.FindLEDCentroids(grayIm)

			im2 = cv2.circle(im, tuple(LED_centroids[0]), 10, (0,0,255), 2)
			im2 = cv2.circle(im2, tuple(LED_centroids[1]), 10, (0,0,255), 2)
			im2 = cv2.circle(im2, tuple(pupilCenter), 10, (0,0,255),2)
			# cv2.imshow('frame', im)
			# cv2.waitKey(1)

		return grayIm

	def getCalibrationPointFromIm(self, im):
		# print "Get Calibration Points from Image"

		# Get Saliency map iffi
		# sm = SaliencyMap(im)
		# print sm
		return [[50,50],[100,100]]
		# oi.imshow_array([sm.map])

	def saveData(self):

		with open("calibration.csv", "wb") as csvFile:
			writer = csv.writer(csvFile, delimiter=',')
			for i in xrange(0,len(self.calibPoints)):
				writer.writerow(np.append(self.calibPoints[i], self.calibPointsLabel[i]))
			# writer.writerow(self.pupilCenters)
			# writer.writerow(self.LED_centroids)


	def optimize(self):
		gazeObject = self.gazeObject();
		# gazeObject.optimizeGaze(self.calibPoints, self.pupilCenters, self.LED_centroids);
		return

class pixelPoint:
	def __init__(self, x, y, radius, color):
		self.x = x
		self.y = y
		self.radius = radius
		self.color = color

class AnimationPanel(wx.Panel):

	def __init__(self, parent):
		wx.Panel.__init__(self, parent)
		self.DataObj = None
		self.camera = None
		self.pointList = []
		self.drawNum = 0
		self.SetBackgroundColour(wx.BLACK)
		self.Bind(wx.EVT_PAINT, self.OnPaint)
		self.timer = wx.Timer(self)
		self.cameraTimer = wx.Timer(self)
		self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
		self.Bind(wx.EVT_TIMER, self.OnCameraTimer, self.cameraTimer)
		self.timer.Start(2000)
		self.cameraTimer.Start(50)

	def setPoint(self, point, radius, color):
		_point = pixelPoint(point[0], point[1], radius, color)
		self.pointList.append(_point)

	def listClear(self):
		self.pointList = []

	def OnPaint(self, event):
		self.dc = wx.PaintDC(self)
		self.listClear();
		self.setDrawData(self.drawNum)
		self.drawPoint()

	def OnTimer(self, event):
		self.drawNum += 1
		if self.drawNum > 9:
			self.camera.saveData()
			self.GetParent().Close()
		self.Refresh()

	def OnCameraTimer(self, event):
		self.camera.update(self.drawNum)


	def drawPoint(self):
		for point in self.pointList:
			self.dc.SetBrush(wx.Brush(point.color, wx.SOLID))
			self.dc.DrawCircle(point.x, point.y, point.radius)

	def setDrawData(self, i):
		target = self.DataObj.calTargetTrain[i]
		self.setPoint(target, 20, "white")

class CalibrationFrame(wx.Frame):
	def __init__(self, parent):
		wx.Frame.__init__(self, parent, wx.ID_ANY, 'Test FullScreen')
	
		self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)
		self.panel = AnimationPanel(self)

	def OnCloseWindow(self, event):
		self.Destroy()

	def OnFullScreen(self):
		self.ShowFullScreen(not self.IsFullScreen(),27)


class CalibrationData:
	def __init__(self, winSize):
		self.calTargetTrain = [];
		self.windowSize = winSize;

		self.setTrainPoint();

	def setTrainPoint(self):
		W = self.windowSize[0]
		H = self.windowSize[1]

		self.calTargetTrain.append([W/2, H/2])
		WList = [0, W/2, W]
		HList = [0, H/2, H]
		offset = [100, 0, -100]

		for i in xrange(0,3):
			x = WList[i] + offset[i];
			for j in xrange(0,3):
				y = HList[j] +offset[j];

				self.calTargetTrain.append([x,y])

		self.calTargetTrain.append([W/2, H/2])

def main():

	app = wx.App()

	gazeObject = GA.gazeObject();
	cameraObject = CameraObject(gazeObject);
	DataObj = CalibrationData(wx.Display(1).GetGeometry().GetSize())

	frame = CalibrationFrame(None)
	# frame.Show(True)
	frame.OnFullScreen()
	frame.panel.DataObj = DataObj
	frame.panel.camera = cameraObject
	app.MainLoop()

	# cap = cv2.VideoCapture(0)

	# while(True):
	# 	ret, frame = cap.read()
	# 	cv2.imshow('frame', frame)
	# 	if cv2.waitKey(1) & 0xFF == ord('q'):
	# 		break

if __name__ == "__main__":
	main();
