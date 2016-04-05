# # import the necessary packages
# import numpy as np
# # import argparse
# import cv2
 
# # construct the argument parser and parse the arguments
# # print "hello"
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# # args = vars(ap.parse_args())

# # # load the image, clone it for output, and then convert it to grayscale
# image = cv2.imread("1.png")
# print image.shape
# image = image[400:600,0:300]

# # print image.shape
# output = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # detect circles in the image
# circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=100, param2=500,minRadius=0,maxRadius=0)
 
# print circles
# # ensure at least some circles were found
# if circles is not None:
# 	# convert the (x, y) coordinates and radius of the circles to integers
# 	circles = np.round(circles[0, :]).astype("int")
 
# 	# loop over the (x, y) coordinates and radius of the circles
# 	for (x, y, r) in circles:
# 		# draw the circle in the output image, then draw a rectangle
# 		# corresponding to the center of the circle
# 		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
# 		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
# # 	# show the output image
# # cv2.imshow("output", np.hstack([image, output]))
# cv2.imshow("output", output)
# cv2.waitKey(0)

import cv2
import numpy as np

# img = np.ones((200,250,3), dtype=np.uint8)
# for i in range(50, 80, 1):
#     for j in range(40, 70, 1):
#         img[i][j]*=200

img = cv2.imread("1.png")
# print image.shape
# img = img[400:600,0:300]

# cv2.circle(img, (180,150), 20, (255,255,255), -1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.5, 20,
              param1=30,
              param2=30,
              minRadius=18,
              maxRadius=25)

print circles
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('circles', img)

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
