import init_path
import rcnnModule
from sklearn import svm
import numpy as np
import os, sys, cv2
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from utils.timer import Timer
from sklearn.externals import joblib

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
	setting['TEST_DIR'] = os.path.join(setting['DATA_DIR'], 'Test')
	setting['DST_DIR'] = os.path.join(setting['DATA_DIR'], 'result')
	setting['DST_MODEL_DIR'] = os.path.join(setting['DST_DIR'], 'imageNet', setting['NET'])
	setting['featureDstDir'] = os.path.join(setting['DST_MODEL_DIR'], "FEATURE")

	categories = sorted([f for f in os.listdir(setting['IMAGE_DIR'])])
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
		imList = np.array(sorted([f for f in os.listdir(dirPath)]))
		imPath = np.array([os.path.join(dirPath, im) for im in imList])
		iid2name = np.append(iid2name, imList)
		iid2path = np.append(iid2path, imPath)
		iid2cid = np.append(iid2cid, np.ones(len(imPath))*cid)
		cid = cid + 1

	iid2cid = iid2cid.astype(int)
	cid2name = np.array(cid2name)
	cid2path = np.array(cid2path)

	return setting, cid2name, cid2path, iid2path, iid2name, iid2cid

def train_SVM(setting, y):
	print "train SVM"
	# SVM Training

	# SVM options
	# svm_kernel                  	= 'rbf';
	# svm_C							= 1.0;
	# svm_loss						= 'squared_hinge'
	# svm_penalty					= 'l2'
	# svm_multi_class				= 'ovr'
	# svm_random_state				= 0 


	filePath = os.path.join(setting['DST_MODEL_DIR'], "svm_trained.pkl")
	try:
		clf = joblib.load(filePath)
		print "using trained model"		
	except:
		print "building svm model"
		X = loadDesc(setting)
		X = X.astype('float')
		timer = Timer()	

		timer.tic()
		clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
		timer.toc()
		print timer.total_time

		joblib.dump(clf, filePath)

	# TEST
	# print clf.decision_function(X[0])
	# print clf.predict(X[5000])
	return clf

def loadDesc(setting):
	print "Load Desc..."
	timer = Timer()	

	featureDstDir = setting['featureDstDir']
	sortedList = sorted([ f for f in os.listdir(featureDstDir)])
	descPath = np.array([ os.path.join(featureDstDir, x) for x in sortedList])

	X = []
	cnt = 0
	size = len(descPath)
	timer.tic()
	for path in descPath:
		feature = readCSV(path)
		X.append(feature)
		print "%d / %d file loaded" % (cnt, size)
		cnt = cnt + 1

	timer.toc()

	# print timer.total_time

	X = np.array(X)
	X = np.reshape(X, X.shape[0:2])
	return X
	
def readCSV(path):
	rlist = []
	with open(path, 'rb') as f:
		reader = csv.reader(f, delimiter=' ')
		for row in reader:
			rlist.append(row)

	return np.array(rlist)

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

def TestModel(setting, rcnnModel, clf):
	print "Test trained Model"
	testDir = setting['TEST_DIR']
	sortedList = sorted([ f for f in os.listdir(testDir)])

	imPath = np.array([ os.path.join(testDir, x) for x in sortedList])
	for path in imPath:
		im  = cv2.imread(path)
		[features, bbox] = rcnnModel.getFeatureIm(im)

		feature = np.mean(features, axis=0)

		predict_result = clf.predict(features)

		print clf.predict(feature)
		print len(np.where(predict_result==0)[0])
	# print imPath

def main():

	[setting, cid2name, cid2path, iid2path, iid2name, iid2cid] = init_train();

	print "rcnnModel loading..."
	rcnnModel = rcnnModule.RcnnObject('zf', False);

	featureExtraction(setting, cid2name, cid2path, iid2path, iid2name, iid2cid, rcnnModel)

	clf = train_SVM(setting, iid2cid)

	TestModel(setting, rcnnModel, clf)

if __name__ == '__main__':
	main()