from System.nms1 import non_max_suppression
import numpy as np
import cv2
# construct a list containing the images that will be examined
# along with their respective bounding boxes
images = [
	("images/raccoon-1.jpg", np.array([
	(12, 30, 76, 94),
	(12, 30, 76, 92),
	(12, 30, 76, 91),
	(12, 30, 76, 90),
	(12, 30, 76, 93),
	(12, 30, 76, 95),
	(12, 36, 76, 100),
	(72, 36, 200, 164),
	(84, 48, 212, 176)]))]
# loop over the images
for (imagePath, boundingBoxes) in images:
	# load the image and clone it
	print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
	print(boundingBoxes)
	image = cv2.imread(imagePath)
	orig = image.copy()
	# loop over the bounding boxes for each image and draw them
	for (startX, startY, endX, endY) in boundingBoxes:
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
	# perform non-maximum suppression on the bounding boxes
	pick = non_max_suppression(boundingBoxes, 0.3)
	print(pick)
	print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
	# loop over the picked bounding boxes and draw them
	for (startX, startY, endX, endY) in pick:
		cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
	# display the images
	cv2.imshow("Original", orig)
	cv2.imshow("After NMS", image)
	cv2.waitKey(0)