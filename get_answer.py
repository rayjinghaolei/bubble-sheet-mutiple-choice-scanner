#import packages
import numpy as np
import imutils
import cv2

# correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

def order_points(pts):
	# 4 coordinate in total
	rect = np.zeros((4, 2), dtype = "float32")

	# from 0 - 3, they are top left, top right, bottom right, bottom left respectively
	# calculate top left , bottom right
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	#calculate top right and bottom left
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# get input coordinates
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# calculate input weight and height
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# new coordinate after the perspective transformation
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# calculate the Matrix for the transformation
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the result
	return warped
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes
def cv_show(name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

# image preprocessing
image = cv2.imread("test_01.png")
contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blurred',blurred)
edged = cv2.Canny(blurred, 75, 200)
cv_show('edged',edged)

# find the contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(contours_img,cnts,-1,(0,0,255),3) 
cv_show('contours_img',contours_img)
docCnt = None

# make sure contours are detected
if len(cnts) > 0:
    # sort the contours by size
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # loop through every contour
    for c in cnts:
        # approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # get ready for perspective transformation
        if len(approx) == 4:
            docCnt = approx
            break

warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('warped',warped)
# OTSU threshold
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)
thresh_Contours = thresh.copy()
# find every circular contour
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)[0]
cv2.drawContours(thresh_Contours,cnts,-1,(0,0,255),3) 
cv_show('thresh_Contours',thresh_Contours)
questionCnts = []

# loop through the circular contours
for c in cnts:
	# calculate the ratio and the size
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# make the standard for question bubble
	if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
		questionCnts.append(c)

# sort from top to down
questionCnts = sort_contours(questionCnts,
	method="top-to-bottom")[0]
correct = 0

# 5 choices every row
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# sort them
	cnts = sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None

	# loop through every bubble
	for (j, c) in enumerate(cnts):
		# get the result using mask
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1) #-1表示填充
		cv_show('mask',mask)
		# see if the answer is chosen by calculating the number of non-zero pixels
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		# check with the threshold
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# check with the correct answer
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# mark as correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# draw the picture
	cv2.drawContours(warped, [cnts[k]], -1, 9, 3)

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)


