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
		self.capEye = cv2.VideoCapture(1)
		self.sceneIm = self.capScene.read()
		self.eyeIm = self.capEye.read()
		self.calibPoints = {}
		self.pupilCenters = {}
		self.LED_centroids = {}
		self.gazeObject = gazeObject

	def update(self):
		self.readFrameScene();
		# self.readFrameEye();

	def readFrameScene(self):
		ret, frame = self.capScene.read()
		# [im, points] = self.imProcessingScene(frame)
		self.sceneIm = frame
		# self.calibPoints[num] = points
		return frame

	def readFrameEye(self):
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

	def imProcessingScene(self, frame):
		points = self.getCalibrationPointFromIm(frame)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		ret,thresh = cv2.threshold(gray,235,255,0)

		if ret == False:
			return gray

		return thresh, points

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
			cv2.imshow('frame', im)
			cv2.waitKey(1)

		return grayIm

	def getCalibrationPointFromIm(self, im):
		# print "Get Calibration Points from Image"

		# Get Saliency map iffi
	    # sm = SaliencyMap(im)
	    # print sm
	    return [[50,50],[100,100]]
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
			self.GetParent().Close()
		self.Refresh()

	def OnCameraTimer(self, event):
		self.camera.update()

		cv2.imshow('frame', self.camera.sceneIm)
		cv2.waitKey(0)

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
