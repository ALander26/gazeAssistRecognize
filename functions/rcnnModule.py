from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, im_feature
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg_cnn_m_1024': ('VGG_CNN_M_1024', 'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
		'caffenet': ('CaffeNet', 'caffenet_fast_rcnn_iter_40000.caffemodel'),
		'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}


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

class RcnnObject:
	def __init__(self, demo_net, cpu_mode):
		# Gaze Object Init
		cfg.TEST.HAS_RPN = True

		prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[demo_net][0], 'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
		caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models', NETS[demo_net][1])

		if not os.path.isfile(caffemodel):
			raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
							'fetch_fast_rcnn_models.sh?').format(caffemodel))
		if cpu_mode:
			caffe.set_mode_cpu()
		else:
			caffe.set_mode_gpu()
			caffe.set_device(0)
		
		self.net = caffe.Net(prototxt, caffemodel, caffe.TEST)

		# Warmup on a dummy image
		im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
		for i in xrange(2):
			_, _= im_detect(self.net, im)

	def getFeatureIm(self, im, gazeData):
		# Fast Rcnn Feature
		obj_proposals = self.getBoundBoxFromEye(im, gazeData)
		feature = im_feature(self.net, im, obj_proposals)

		return feature

	def getBoundBoxFromEye(self, im, gazeData):
		# Get object box from gaze data
		print "hi"
		box, score = self.getObjectnessBoxBing(im)

	def getFeatureIm(self, im):
		# timer = Timer()
		# timer.tic()
		
		features, boxes = im_feature(self.net, im)
		# timer.toc()

		# print features.shape
		return features, boxes
		# Visualize detections for each class
		# CONF_THRESH = 0.8
		# NMS_THRESH = 0.3
		# for cls_ind, cls in enumerate(CLASSES[1:]):
		# 	cls_ind += 1 # because we skipped background
		# 	cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		# 	cls_scores = scores[:, cls_ind]
		# 	dets = np.hstack((cls_boxes,
		# 					  cls_scores[:, np.newaxis])).astype(np.float32)
		# 	keep = nms(dets, NMS_THRESH)
		# 	dets = dets[keep, :]
		# 	vis_detections(im, cls, dets, thresh=CONF_THRESH)

# test = RcnnObject('zf', False)

# test.demoTest('000456.jpg')
# print cfg.DATA_DIR