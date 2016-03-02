from fast_rcnn.config import cfg
from fast_rcnn.test import im_feature
from utils.cython_nms import nms
from utils.timer import Timer
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

NETS = {'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

class RcnnObject:
	def __init__(self, demo_net, cpu_mode):
		# Gaze Object Init

		prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[demo_net][0],
						'test.prototxt')
		caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models', NETS[demo_net][1])

		if not os.path.isfile(caffemodel):
			raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
							'fetch_fast_rcnn_models.sh?').format(caffemodel))
		if cpu_mode:
			caffe.set_mode_cpu()
		else:
			caffe.set_mode_gpu()
			caffe.set_device(args.gpu_id)
		
		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	def getFeatureIm(self, im, gazeData):
		# Fast Rcnn Feature
		obj_proposals = self.getBoundBoxFromEye(im, gazeData)
		feature = im_feature(net, im, obj_proposals)

		return feature

	def getBoundBoxFromEye(self, im, gazeData):
		# Get object box from gaze data
		print "hi"
		box, score = self.getObjectnessBoxBing(im)

	def getObjectnessBoxBing(self, im):
		# b = Bing(~~~)
		box, score = b.predict(im)

		return box, score

