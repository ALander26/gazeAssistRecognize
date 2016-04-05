import sys
import init_path
import gaze_algorithm as GA
import numpy as np
import cv2, os
import wx
import time

class CameraObject():
	def __init__(self, gazeObject):
		self.capScene = cv2.VideoCapture(0)
		# self.capEye = cv2.VideoCapture(1)
		self.sceneIm = self.capScene.read()
		self.sceneThresh = None
		# self.eyeIm = self.capEye.read()
		self.calibPoints = []
		self.calibPointsLabel = []
		self.pupilCenters = []
		self.LED_centroids = []
		self.gazeObject = gazeObject

	def update(self, drawLabel):
		points = self.readFrameScene();
		pupilCenter, LED_centroid = self.readFrameEye();

		self.calibPoints.append(points)
		self.calibPointsLabel.append(drawLable)
		self.pupilCenters.append(pupilCenter)
		self.LED_centroids.append(LED_centroid)

	def readFrameScene(self):
		scene_threshold = 232
		ret, frame = self.capScene.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, scene_threshold ,255,0)

		points = self.imProcessingScene(thresh)
		self.sceneIm = frame
		self.sceneThresh = thresh
		if points is None:
			return frame


		return frame

	def readFrameEye(self):
		ret, frame = self.capEye.read()
		[frame, pupilCenter, LED_centroid] = self.imProcessingEye(frame)

		if pupilCenter == None:
			return None, None, self.eyeIm

		# self.pupilCenters[drawLabel] = pupilCenter
		# self.LED_centroids[drawLabel] = LED_centroids
		self.eyeIm = frame
		return pupilCenter, LED_centroid

	def clearObject(self):
		self.capScene.release()
		# self.capEye.release()
		cv2.destroyAllWindows()

	def imProcessingScene(self, frame):
		# points = self.getCalibrationPointFromIm(frame)
		# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		output = frame.copy()
		circles = cv2.HoughCircles(output, cv2.cv.CV_HOUGH_GRADIENT, 1.1, 2000,
			param1=30,
			param2=5,
			minRadius=5,
			maxRadius=8)

		if circles is None:
			return None

		points = circles[0,0]
		# out = self.camera.sceneIm.copy()
		# ensure at least some circles were found
		# print circles[0,0]
		# circles = np.uint16(np.around(circles))
		# for i in circles[0,:]:
		# 	cv2.circle(out,(i[0],i[1]),i[2],(0,255,0),2)
		# 	cv2.circle(out,(i[0],i[1]),2,(0,0,255),3)
		
		# cv2.imshow("output", out)
		# cv2.waitKey(0)


		# points = [0,0]
		# resized_image = cv2.resize(frame, (500, 375)) 
		# gray = resized_image
		return points

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
		self.cameraTimer.Start(20)

		# self.video = cv2.VideoWriter('video.avi',cv2.cv.CV_FOURCC(*'XVID'), 20, (640,480))

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
			self.camera.clear()
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
	frame.OnFullScreen()
	frame.panel.DataObj = DataObj
	frame.panel.camera = cameraObject
	app.MainLoop()

if __name__ == "__main__":
	main();
