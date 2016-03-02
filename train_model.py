import init_path
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

def init_train():
	setting = {}
	setting['ROOT_DIR'] = os.getcwd()
	setting['DATA_DIR'] = os.path.join(setting['ROOT_DIR'], 'data')
	setting['DST_DIR'] = os.path.join(setting['DATA_DIR'], 'result')
	categories = os.listdir(setting['DATA_DIR'] + '\scene')
	categoryDirPath = [os.path.join(setting['DATA_DIR'], 'scene', f) for f in categories]

	cid2name = categories
	cid2path = categoryDirPath
	iid2path = np.array([])
	iid2name = np.array([])
	iid2cid = np.array([])

	cid = 0
	for dirPath in categoryDirPath:
		imList = np.array(os.listdir(dirPath))
		imPath = np.array([os.path.join(dirPath, im) for im in imList])
		iid2name = np.append(iid2name, imList)
		iid2path = np.append(iid2path, imPath)
		iid2cid = np.append(iid2cid, np.ones(len(imPath))*cid)
		cid = cid + 1

	cid2name = np.array(cid2name)
	cid2path = np.array(cid2path)

	return setting, cid2name, cid2path, iid2path, iid2name, iid2cid

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg_cnn_m_1024')

    args = parser.parse_args()

    return args

def train_SVM(idx2desc, cid, iid2cid):
	print "train SVM"
	# SVM Training

	numDesc = len(idx2desc)

	target = np.array([])
	for i in xrange(0,numDesc):
		imageID = iid2cid(i)
		if imageID == cid:
			target = np.append(target, np.array([1]))
		else:
			target = np.append(target, np.array([-1]))


	imageFeatures = idx2desc
	clf = svm.LinearSVC(C=10)
	clf.fit(imageFeatures, target)

	w = clf.get_params()
	return clf

def main():

	args = parse_args()

	prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
					'test.prototxt')
	caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models', NETS[args.demo_net][1])

	if not os.path.isfile(caffemodel):
		raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
						'fetch_fast_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	[setting, cid2name, cid2path, iid2path, iid2name, iid2cid] = init_train();

	featureDstDir = os.path.join(setting['DST_DIR'], 'SCENE_FEATURE')
	if os.path.exists(featureDstDir):
		print ""
	else:
		os.mkdir(featureDstDir)
	# len(iid2path)
	numIm = 7
	for i in range(0, numIm):
		fileName = "ID%06d.mat" % (i+1)
		box_file = os.path.join(setting['DST_DIR'], 'SCENE', fileName)
		obj_proposals = np.array(sio.loadmat(box_file)['desc'])
		obj_proposals = obj_proposals[:,0:4].astype(int)
		im = cv2.imread(iid2path[i])

		feature = im_feature(net, im, obj_proposals)

		filePath = os.path.join(featureDstDir, fileName)
		sio.savemat(filePath, {'feature':feature})
		print "%.2f percent complete" % ((i+1)*100/float(numIm))



# setting.svm.kernel                  = 'NONE';               % DO NOT TOUCH) Additive kernel map before training SVMs.
# setting.svm.norm                    = 'L2';                 % DO NOT TOUCH) Normalization type before training SVMs.
# setting.svm.c                       = 10;                   % DO NOT TOUCH) SVM parameter.
# setting.svm.epsilon                 = 1e-3;                 % DO NOT TOUCH) SVM parameter.
# setting.svm.biasMultiplier          = 1;                    % DO NOT TOUCH) SVM parameter.
# setting.svm.biasLearningRate        = 0.5;                  % DO NOT TOUCH) SVM parameter.
# setting.svm.loss                    = 'HINGE';              % DO NOT TOUCH) SVM parameter.
# setting.svm.solver                  = 'SDCA';               % DO NOT TOUCH) SVM parameter.



if __name__ == '__main__':
	main()