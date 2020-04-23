# Usage:
# python3 people_tracker.py -i street.mp4
# -i : input file name

from trackers.centroidtracker import CentroidTracker
from trackers.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", 
	default="MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",
	default="MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str, default="test.mp4",
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, default="output/result.avi",
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
args = vars(ap.parse_args())


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


print("[INFO] opening video file...")
vs = cv2.VideoCapture(args["input"])

writer = None

W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# stop when the video ends
	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
	status = "Waiting"
	rects = []

	# check what algorithm to run
	# if frame number is a multiple of skip frames run detection
	if totalFrames % args["skip_frames"] == 0:
		if totalFrames % (args["skip_frames"] * 10) == 0:
			print("Detecting", str(totalFrames))
		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum confidence
			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue
				
				# compute the (x, y)-coordinates of the bounding box
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)
				trackers.append(tracker)
		
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"
			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))
			# draw rectangle for person
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
	
	objects = ct.update(rects)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current objectID
		to = trackableObjects.get(objectID, None)
		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to
		# draw both the ID of the object to the frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	if writer is not None:
		writer.write(frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	totalFrames += 1
	fps.update()
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
if writer is not None:
	writer.release()

else:
	vs.release()
	cv2.destroyAllWindows()

					