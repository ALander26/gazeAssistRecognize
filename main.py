import init_path
import gaze_algorithm as GA
# import train_model as TM
# import rcnnModule
import numpy as np
import cv2, os, sys
import time
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
# from saliency_map import SaliencyMap
# from utils import OpencvIo

CLASSES = ('__background__',
           'book', 'cellphone', 'cloud', 'coffee',
           'computer', 'cup', 'man', 'mouse', 'pen',
           'people', 'road', 'woman')

_WINSIZE = 5

class CameraObject():
	def __init__(self, gazeObject):
		self.capScene = cv2.VideoCapture(1)
		self.capEye = cv2.VideoCapture(0)
		self.sceneIm = self.capScene.read()
		self.eyeIm = self.capEye.read()
		self.calibPoints = {}
		self.pupilCenters = {}
		self.LED_centroids = {}
		self.gazeObject = gazeObject

	def update(self, num):
		self.readFrameScene(num);
		self.readFrameEye(num);

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

# def threadFunc(image, gazeData):
# 	# print "aaaa"
# 	feature = rcnnObject.getFeatureIm(image, gazeData);


def loadTrainModel():
	setting = {}
	setting['NET'] = 'zf'
	setting['ROOT_DIR'] = os.getcwd()
	setting['DATA_DIR'] = os.path.join(setting['ROOT_DIR'], 'data')
	setting['DST_DIR'] = os.path.join(setting['DATA_DIR'], 'result')	
	setting['DST_MODEL_DIR'] = os.path.join(setting['DST_DIR'], 'imageNet', setting['NET'])
	filePath = os.path.join(setting['DST_MODEL_DIR'], "svm_trained.pkl")

	clf = joblib.load(filePath)

	return clf

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def main():
	# rcnnModel = rcnnModule.RcnnObject('zf', False);
	gazeObject = GA.gazeObject();
	cam = CameraObject(gazeObject);

	# clf = loadTrainModel()

	while(True):
		cam.update(0)
		cv2.imshow('frame', cam.eyeIm)
		cv2.waitKey(1)
		# [feature, boxes] = rcnnModel.getFeatureIm(cam.sceneIm)

		# predict_result = clf.predict(feature)

		# print len(np.where(predict_result == 7)[0])
		# # vis_detection(cam.sceneIm, )
		# feature_mean = np.mean(feature, axis=0)

		# print clf.predict(feature_mean), clf.decision_function(feature_mean)
		# bookNum = np.where(List == 0)[0]

if __name__ == '__main__':
	main();