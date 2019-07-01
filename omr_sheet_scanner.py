import imutils
import cv2 as cv
import numpy as np
from imutils import contours
from argparse import ArgumentParser
from imutils.perspective import four_point_transform


#initialise argument parser and add variable
ap = ArgumentParser()
ap.add_argument("--image", type=str, default="example_test.png")
args = vars(ap.parse_args())

#define answer key to omr sheet
answer_key = {0:1, 1:4, 2:0, 3:3, 4:1}

#load image, convert it to grayscale
#blur it and find edges
image = cv.imread(args["image"])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blured = cv.GaussianBlur(gray, (5,5), 0)
edged = cv.Canny(blured, 75, 200)

#find contours
cnts = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#sort contours on basis of their area
cnts = sorted(cnts, key=cv.contourArea, reverse=True)
doc_cnt = None
for c in cnts:
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02*peri, True)

	#check if there are four points
	if len(approx) == 4:
		doc_cnt = approx
		break

#draw contour on image
cv.drawContours(image.copy(), [doc_cnt], 0, (0, 255, 0), 2)

#apply four point transform to get birds eye view
paper = four_point_transform(gray, doc_cnt.reshape(4, 2))
paper_colored = four_point_transform(image, doc_cnt.reshape(4, 2))

#apply threshold
thresh = cv.threshold(paper, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)[1]

#run counters over the paper and find all bubbles
cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, 
					   cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
question_cnts = []
copy = paper_colored.copy()
#loop over all contours to mark bubbles area as questions
for c in cnts:
	x,y,w,h = cv.boundingRect(c)
	ar = float(w) / float(h)

	if (w >= 20) and  (h >= 20) and (ar <= 1.1):
		question_cnts.append(c)
		#draw contour on image
		circles = cv.drawContours(copy, [c], 0, (0, 0, 255), 2)

#sort questions from top to bottom
questionCnts = contours.sort_contours(question_cnts, method="top-to-bottom")[0]
correct = 0

#loop over every question and sort options from left to right
# question in batches of 5

answer = paper_colored.copy()

for (q, i) in enumerate(np.arange(0, len(question_cnts), 5)):
	cnts = contours.sort_contours(question_cnts[i:i + 5])[0]
	bubbled = None
	
	#loop over sorted contours
	for (j,c) in enumerate(cnts):
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv.drawContours(mask, [c], -1, 255, -1)

		#count non zero pixel in contour
		mask = cv.bitwise_and(thresh, thresh, mask=mask)
		total = cv.countNonZero(mask)

		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	#check if select bubble is correct or not
	color = (0,0,255)		
	k = answer_key[q]
	print(bubbled)

	if k == bubbled[1]:
		color = (0,255,0)
		correct += 1

	# draw the outline of the correct answer
	answer = cv.drawContours(answer, [cnts[k]], -1, color, 3)

#show score
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv.putText(answer, "{:.2f}%".format(score), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
		   0.9, (0, 0, 255), 2)

cv.imshow("Paper", paper_colored)
cv.imshow("Answer", answer)
cv.waitKey(0)
