import init_path
import rcnnModule
from sklearn import svm
import numpy as np
import os, sys, cv2
import csv

CLASSES = ('__background__',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat', 'chair',
		   'cow', 'diningtable', 'dog', 'horse',
		   'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg_cnn_m_1024': ('VGG_CNN_M_1024', 'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
		'caffenet': ('CaffeNet', 'caffenet_fast_rcnn_iter_40000.caffemodel'),
		'zf': ('ZF', 'ZF_faster_rcnn_final.caffemodel')}


def init_train():
	print "Init Train..."
	setting = {}
	setting['NET'] = 'zf'
	setting['ROOT_DIR'] = os.getcwd()
	setting['DATA_DIR'] = os.path.join(setting['ROOT_DIR'], 'data')
	setting['IMAGE_DIR'] = os.path.join(setting['DATA_DIR'], 'imageNet', 'images')
	setting['DST_DIR'] = os.path.join(setting['DATA_DIR'], 'result')
	setting['featureDstDir'] = os.path.join(setting['DST_DIR'], 'imageNet', setting['NET'], "FEATURE")
	categories = os.listdir(setting['IMAGE_DIR'])
	categoryDirPath = [os.path.join(setting['IMAGE_DIR'], f) for f in categories]

	cid2name = categories
	cid2path = categoryDirPath
	iid2path = np.array([])
	iid2name = np.array([])
	iid2cid = np.array([])

	cNum = len(cid2path)
	cid = 0
	for dirPath in categoryDirPath:
		# dirPath = cid2path[i]
		imList = np.array(os.listdir(dirPath))
		imPath = np.array([os.path.join(dirPath, im) for im in imList])
		iid2name = np.append(iid2name, imList)
		iid2path = np.append(iid2path, imPath)
		iid2cid = np.append(iid2cid, np.ones(len(imPath))*cid)
		cid = cid + 1

	iid2cid = iid2cid.astype(int)
	cid2name = np.array(cid2name)
	cid2path = np.array(cid2path)

	return setting, cid2name, cid2path, iid2path, iid2name, iid2cid

def train_SVM(setting, idx2desc, iid2cid):
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

def loadDesc(setting, iid2path):
	print "Load Desc..."
	featureDstDir = setting['featureDstDir']

def readCSV(path):
	with open(path, 'rb') as f:
		reader = csv.reader(f, )

def writeCSV(data, path):
	with open(path, 'wb') as fout:
	    writer = csv.writer(fout, delimiter=',')
	    for d in data:
	    	writer.writerow([d])

def featureExtraction(setting, cid2name, cid2path, iid2path, iid2name, iid2cid, rcnnModel):
	print "Feature Extraction.."
	featureDstDir = setting['featureDstDir']

	if not os.path.exists(featureDstDir):
		os.makedirs(featureDstDir)

	numIm = len(iid2path)

	descExist = np.zeros(numIm)
	fList = np.array([ int(x[0:-4]) for x in os.listdir(featureDstDir) ])

	for i in fList:
		descExist[i] = 1

	nonDescList = np.where(descExist == 0)[0]
	numDesc = len(nonDescList)

	if numDesc==0:
		print "No image to desc."

	cnt = 0
	for i in nonDescList:
		print i, cid2name[iid2cid[i]], iid2name[i],": %0.2f percent finished" % (cnt*100.0/numDesc)
		im  = cv2.imread(iid2path[i])
		[features, bbox] = rcnnModel.getFeatureIm(im)

		feature = np.mean(features, axis=0)

		fileName = "%06d.csv" % i
		filePath = os.path.join(featureDstDir, fileName)
		writeCSV(feature, filePath)
		cnt = cnt+1

def main():

	[setting, cid2name, cid2path, iid2path, iid2name, iid2cid] = init_train();

	print "rcnnModel loading..."
	rcnnModel = rcnnModule.RcnnObject('vgg_cnn_m_1024', False);

	featureExtraction(setting, cid2name, cid2path, iid2path, iid2name, iid2cid, rcnnModel)

	idx2desc = loadDesc(setting, iid2path)
	train_SVM(setting, idx2desc, iid2cid)
	# print init_train()


	# for i in xrange(0, numCls):
	# 	Dir = os.path.join(featureDstDir, cid2name[i])
	# 	if not os.path.exists(clsDir):
	# 		os.mkdir(clsDir)

	# numIm = 1


		# print bbox.shape, feature.shape
		# print type(iid2cid[i]), cid2name[iid2cid[i]], iid2name[i]
		# print feature.shape

	# for i in xrange(0, numCls):
	# 	for j
	# idx2desc = featureExtraction(setting, iid2path, iid2cid)

	# numIm = 7
	# for i in range(0, numIm):
	# 	fileName = "ID%06d.mat" % (i+1)
	# 	box_file = os.path.join(setting['DST_DIR'], 'SCENE', fileName)
	# 	obj_proposals = np.array(sio.loadmat(box_file)['desc'])
	# 	obj_proposals = obj_proposals[:,0:4].astype(int)
	# 	im = cv2.imread(iid2path[i])

	# 	feature = im_feature(net, im, obj_proposals)

	# 	filePath = os.path.join(featureDstDir, fileName)
	# 	sio.savemat(filePath, {'feature':feature})
	# 	print "%.2f percent complete" % ((i+1)*100/float(numIm))



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