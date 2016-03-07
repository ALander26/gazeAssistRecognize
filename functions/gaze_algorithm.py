import sys, os
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import leastsq, minimize
from skimage.measure import regionprops, label
from skimage import morphology
from skimage import segmentation
from numpy.linalg import eig, inv
from scipy.spatial.distance import euclidean

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

class gazeObject:
	def __init__(self):
		print "Gaze Object Init"
		self.eye_camera_K_inv = inv(np.array([[940.8864,0,313.0809],[0,937.1089,276.4065],[0,0,1]]))
		self.scene_camera_K_inv = inv(np.array([[1425.91033218970,0,648.841876839901],[0,1429.73734710815,487.002267985083],[0,0,1]]))
		self.LED_OFFSET = np.array([0,0])
		self.pupil_OFFSET = np.array([0,0])
		self.led_OFFSET = np.array([0,0])
		self.pupil_BOX = np.array([0,0,0,0])
		self.pupil_thresh = 47
		self.reflection_thresh = 190
		self.winsize = 5
		self.led_world_positions = np.array([[6.3640,6.3640],[-6.3640,6.3640],[0,0]])
		self.system_params = np.array([0,0,0,0,0,0,0,0,0,0])
		self.distance_to_eye = 52
		self.pupil_image_center = None
		self.ledCentroids = None
		self.corneal_radius = 7.7
		self.corneal_center = None
		self.calibPoints = None
		self.distance_from_C_to_P = 4.5
		self.n1 = 1.3375
		self.old_C = np.array([0,0,0])
		self.old_P = np.array([0,0,0])
		self.system_params = np.array([-18, 100, 70, 0, 0, 0, 130, 7.7, 5.0, 1.5])

	def GetPupilBoundaryPoints(self, im):
		print "Get pupil boundary points."

		ret, bw_im = cv2.threshold(im, self.pupil_thresh, 255, cv2.THRESH_BINARY)
		# ret, reflection_im = cv2.threshold(im, self.reflection_thresh, 255, cv2.THRESH_BINARY)

		# cv2.imshow('frame', bw_im)
		# cv2.waitKey(0)	

		roi_im = self.setROI(bw_im)

		laplacian = cv2.Laplacian(roi_im, cv2.CV_64F)

		inds = np.where(laplacian > 0)

		inds = np.array(inds)

		if len(inds) < 2:
			return None # blink

		inds = inds.T
		inds = inds[:,0:2]
		hull = ConvexHull(inds)
		# for simplex in hull.simplices:
		# 	# plt.plot(inds[simplex, 0], inds[simplex, 1], 'k-')
		# 	# plt.plot(inds[hull.vertices,0], inds[hull.vertices,1], 'r--', lw=2)
		# 	plt.plot(inds[hull.vertices[0],0], inds[hull.vertices[0],1], 'ro')

		inds = inds[hull.vertices, :]
		# print inds
		# plt.plot(inds[:,0], inds[:,1], 'o')
		# plt.show()
		pupil_Center = self.fitEllipse(inds[:,0], inds[:,1])

		return pupil_Center


	def FindCornealCenter(self, led_image_position, distance, corneal_radius):
		print "Find cornel center."
		led_world_positions = self.led_world_positions.T
		led_image_position = np.concatenate((led_image_position, np.array([[1,1]]).T), axis=1)
		reflection_vec = self.eye_camera_K_inv.dot(led_image_position.T).T

		reflection_vec[0] = (reflection_vec[0] / np.linalg.norm(reflection_vec[0])) * distance
		reflection_vec[1] = (reflection_vec[1] / np.linalg.norm(reflection_vec[1])) * distance

		b1 = np.cross(led_world_positions[0], reflection_vec[0]);
		b2 = np.cross(led_world_positions[1], reflection_vec[1]);
		b = np.cross(b1, b2);
		b_norm = b / np.linalg.norm(b)

		corneal_center = (corneal_radius + distance) * b_norm;
		self.corneal_center = corneal_center

		return corneal_center

	def FindLEDCentroids(self, im):
		print "Find LED Centroids."

		roi_im = self.setROILED(im)

		cv2.imshow('frame', roi_im)
		cv2.waitKey(0)	

		ret, bw_im = cv2.threshold(roi_im, self.reflection_thresh, 255, cv2.THRESH_BINARY)

		cleand = morphology.remove_small_objects(bw_im, min_size=36, connectivity=2)
		label_img = label(cleand, connectivity=cleand.ndim)


		props = regionprops(label_img)

		centroids = np.zeros((2,2))
		# OFFSET = np.array([self.OFFSET_X, self.OFFSET_Y])
		centroids[0] = props[0].centroid + self.LED_OFFSET
		centroids[1] = props[1].centroid + self.LED_OFFSET

		print centroids

		centroids = np.fliplr(centroids)
		centroids = centroids.astype(int)

		centroids = centroids + self.led_OFFSET
		
		if centroids[0][0] < centroids[1][0]:
			temp = centroids[1].copy()
			centroids[1] = centroids[0]
			centroids[0] = temp
		
		print centroids
		self.ledCentroids = centroids
		return centroids

	def FindCalibrationPointsGBVS(self):
		print "Find calibration points using GBVS"

	def FindPupilCenter(self, C, rj, alpha, beta):
		print "Find pupil center."
		transformed_C = C-C
		transformed_rj = rj - C
		transformed_O = -C

		x_hat = -transformed_O
		x_hat = x_hat / np.linalg.norm(x_hat)
		z_hat = np.cross(x_hat, transformed_O - transformed_rj)
		z_hat = z_hat / np.linalg.norm(z_hat)
		y_hat = np.cross(z_hat, x_hat)
		y_hat = y_hat / np.linalg.norm(y_hat)

		rotation_matrix = np.array([x_hat, y_hat, z_hat])

		transformed_C = rotation_matrix.dot(transformed_C)
		transformed_rj = rotation_matrix.dot(transformed_rj)
		transformed_O = rotation_matrix.dot(transformed_O)		

		# print transformed_C, transformed_rj, transformed_O
		r = self.distance_from_C_to_P
		n1 = self.n1
		best_error = 999999999
		best_pt = None
		best_theta = None

		for i in range(0,1000):
			theta = (np.pi*2*i)/1000
			pt = [r*np.cos(theta), r*np.sin(theta), 0]
			lhs = n1 * np.dot(np.linalg.norm(np.cross((transformed_rj - transformed_C), (transformed_rj - pt))), np.linalg.norm(transformed_O - transformed_rj))
			rhs = np.dot(np.linalg.norm(np.cross((transformed_rj - transformed_C), (transformed_O - transformed_rj))), np.linalg.norm(pt - transformed_rj))
			
			error = abs(lhs-rhs)
			if error < best_error:
				best_error = error
				best_pt = pt
				best_theta = theta

		# print best_error, best_pt, best_theta
		P_optic = best_pt
		vec = P_optic / np.linalg.norm(P_optic)

		phi_eye = asind(vec[1])
		theta_eye = asind(vec[0] / cosd(phi_eye))

		gaze_adjustment_vec = np.array([cosd(phi_eye+beta)*sind(theta_eye+alpha), sind(phi_eye+beta), -cosd(phi_eye+beta)*cosd(theta_eye+alpha)])
		P_visual = r * gaze_adjustment_vec
		# print P_visual
		P = inv(rotation_matrix).dot(P_visual)
		# print P

		P = P + C
		return P

	def FindPointOfRefraction(self, distance):
		print "Find points of refracation."

		pupil_image_center = self.pupil_image_center
		pupil_image_center = np.concatenate((pupil_image_center,[1]))

		pupil_reflection = self.eye_camera_K_inv.dot(pupil_image_center)

		pupil_reflection = pupil_reflection / np.linalg.norm(pupil_reflection)

		pupil_reflection = pupil_reflection * distance

		return pupil_reflection
		
	def fitEllipse(self, x,y):
		print "Fit Ellipse"

		pupil = cv2.fitEllipse(np.array([x, y]).T)

		self.pupil_image_center = (np.array(pupil[0]) + self.pupil_OFFSET).astype(int)

		return self.pupil_image_center

	def FindCalibPoitns(self, im):
		print "Find calibration points"

	def setROI(self, im):
		im1 = im.copy()

		# cv2.imwrite('01gray.jpg',im)
		# ret,thresh = cv2.threshold(im,55,255,0)

		# cv2.imwrite('02thresh.jpg',im1)
		im2, contours = cv2.findContours(im1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		print im2, contours

		return im

		# maxArea = 0;
		# maxContour = None;

		# for cnt in contours:
		# 	# print cv2.contourArea(cnt)
		# 	if cv2.contourArea(cnt) < 20000 and cv2.contourArea(cnt) > 3000 and cv2.contourArea(cnt) > maxArea:
		# 		maxArea = cv2.contourArea(cnt)
		# 		maxContour = cnt

		# contours = maxContour
		# x,y,w,h = [x + y for x, y in zip(list(cv2.boundingRect(contours)), [-10, -10, 20, 20])]

		# self.pupil_OFFSET = np.array([x,y])
		# self.pupil_BOX = np.array([x,y,w,h])
		# im2 = im[y:y+h, x:x+w]

		# return im2


	def setROILED(self, im):
		im1 = im.copy()

		box = self.pupil_BOX

		x,y,w,h = box + [-20,-30, 40, 60]
		im2 = im[y:y+h, x:x+w]

		self.led_OFFSET = np.array([x,y])
		return im2

	def optimizeGaze(self, calib_points, pupil_centers, led_centroids):
		print "Optimize Gaze Point"
		system_params = self.system_params
		result = leastsq(self.eyeProcessLinCost, system_params, args=(calib_points, pupil_centers, led_centroids))

		return result[0]

	def eyeProcess_advanced(self):
		print "eyeProcess advanced"
	def eyeProcess(self, system_params, pupil_center, centroids, num, last_C, last_P):
		print "Eye process"
		eye_K_inv = self.eye_camera_K_inv
		scene_K_inv = self.scene_camera_K_inv

		scenecam_tx = system_params[0]
		scenecam_ty = system_params[1]
		scenecam_tz = system_params[2]
		scenecam_rx = system_params[3]
		scenecam_ry = system_params[4]
		scenecam_rz = system_params[5]
		distance = system_params[6]
		corneal_radius = system_params[7]
		alpha_eye = system_params[8]
		beta_eye = system_params[9]

		distance_from_C_to_P = 4.5
		n1 = 1.3375

		Rx = [[1,0,0],[0,cosd(scenecam_rx), -sind(scenecam_rx)], [0, sind(scenecam_rx), cosd(scenecam_rx)]]
		Ry = [[cosd(scenecam_ry), 0, sind(scenecam_ry)], [0,1,0], [-sind(scenecam_ry), 0, cosd(scenecam_ry)]]
		Rz = [[cosd(scenecam_rz), -sind(scenecam_rz), 0], [sind(scenecam_rz), cosd(scenecam_rz), 0], [0,0,1]]

		# reflection_world = K.dot(centroids)
		# reflection_world[0] = reflection_world[0] / np.linalg.norm(reflection_world[0])
		# reflection_world[1] = reflection_world[1] / np.linalg.norm(reflection_world[1])
		# reflection_world = reflection_world * distance

		C = self.FindCornealCenter(centroids, distance, corneal_radius);
		pupil_reflection = self.FindPointOfRefraction(distance);
		P = self.FindPupilCenter(C, pupil_reflection, alpha_eye, beta_eye)

		# C = self.FindCornealCenter(led_positions, reflection_world, corneal_radius, distance_to_eye);
		# pupil_reflection = self.FindPointOfRefraction(K, image_pupil_center, distance_to_eye, OFFSET_X, OFFSET_Y);
		# P = self.FindPupilCenter(C, pupil_reflection, alpha_eye, beta_eye);

		# ic = K.dot(C)
		# ic = ic / ic[2]
		# ip = K.dot(P)
		# ip = ip / ip[2]

		if num > 1:
			C = C * 0.5 + last_C * 0.5
			P = P * 0.5 + last_P * 0.5

		# C_out = C
		# P_out = P

		C = C - [scenecam_tx, scenecam_ty, scenecam_tz]
		P = P - [scenecam_tx, scenecam_ty, scenecam_tz]

		Cf = np.dot(Rz,np.dot(Ry,np.dot(Rx,C)))
		Pf = np.dot(Rz,np.dot(Ry,np.dot(Rx,P)))

		c_to_p_unit_vec = Pf - Cf
		P_inf = Cf + c_to_p_unit_vec * 10000

		scale_vec = (1 - Cf[2])/c_to_p_unit_vec[2]

		perspective_point = Cf + scale_vec*c_to_p_unit_vec

		ep = inv(scene_K_inv).dot(perspective_point)
		sc = inv(scene_K_inv).dot(Cf)
		sc = sc / sc[2]
		sp = inv(scene_K_inv).dot(Pf)
		sp = sp / sp[2]
		sp_inf = inv(scene_K_inv).dot(P_inf)
		sp_inf = sp_inf / sp_inf[2]

		gaze_slope = (sc[1]-sp[1])/ (sc[0] - sp[0])
		gaze_offset = sp(1) - gaze_slope*sp[0]

		return ep, sc, sp, sp_inf, gaze_slope, gaze_offset

	def eyeProcessLinCost(self, system_params, calib_points, pupil_centers, led_centroids):
		print "Process Cost function for least square"

		old_C = self.old_C
		old_P = self.old_P

		size = len(calib_points)
		d = np.zeros(size)
		for i in range(0,size):
			[ep, old_C, old_P, sc, sp, sp_inf, slope, offset] = self.eyeProcess(system_params, pupil_centers[i], led_centroids[i], old_C, old_P);			

			d[i] = math.sqrt((calib_points[i][0] - ep[0])^2 + (calib_points[i][1] - ep[1])^2)

		self.old_C = old_C
		self.old_P = old_P

		return d

def cosd(degree):
	radian = degree * np.pi / 180;
	return np.cos(radian)

def sind(degree):
	radian = degree * np.pi / 180;
	return np.sin(radian)

def asind(value):
	radian = np.arcsin(value)
	return radian * 180 / np.pi

def acosd(value):
	radian = np.arccos(value)
	return radian * 180 / np.pi
