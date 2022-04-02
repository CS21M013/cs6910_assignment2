import cv2
import numpy as np 
import argparse
import time

# Parsing the argument deciding whether real time detection needs to be performed, detecttion in a video, or in a image needs to be performed
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False) #detection from webcam
parser.add_argument('--play_video', help="Tue/False", default=False) # detection from video
parser.add_argument('--image', help="Tue/False", default=False) # detection from image
parser.add_argument('--video_path', help="Path of video file", default="videos/fire1.mp4") # the path of the video to load the video for detection
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg") # path of the image file
parser.add_argument('--verbose', help="To print statements", default=True)
# parse the arguments
args = parser.parse_args()


'''
Load yolo-v3 - Basically pre trained weights have been loaded that have been trained to detect guns, rifles and fires.
'''
def load_yolo():
	# load the weights
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("obj.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
    # return the layers and the network and the classes    
	layers_names = net.getLayerNames()
	output_layers = [layers_names[i-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

'''
Load the image file
'''
def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

'''
Start the webcam for detection in webcam
'''
def start_webcam():
	cap = cv2.VideoCapture(0)

	return cap

'''
Displaying the images in three channels (RGB)
'''
def display_blob(blob):
	# Three images each for RED, GREEN, BLUE channel
	for b in blob:
		for n, imgb in enumerate(b):
			cv2.imshow(str(n), imgb)
'''
Function to detect the objects
'''
def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

'''
Function to generate boxes around the detected object
'''
def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.2:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
			
'''
Function to draw labels on the boxes detecting the guns, rifles and fires
'''
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[0]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	img=cv2.resize(img, (1980,1080))
	cv2.imshow("Image", img)

'''
Function to detect the images
'''
def image_detect(img_path): 
    # load the model,classes,and the output_layers of the pre-trained model
	model, classes, colors, output_layers = load_yolo()
    # loading the image
	image, height, width, channels = load_image(img_path)
    # detecting the objects
	blob, outputs = detect_objects(image, model, output_layers)
    # generate the bozes around the detected image
	boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    # draw the labels around the boxes
	draw_labels(boxes, confs, colors, class_ids, classes, image)
	while True:
		key = cv2.waitKey(1)
		if key == 27:
			break

'''
Function to detect using webcam
'''
def webcam_detect():
	# load the pre-trained model
    model, classes, colors, output_layers = load_yolo()
	# starting the webcam
    cap = start_webcam()
	while True:
		# read singe frame of the webcam
        _, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
		# detect the images
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        # Drawing the labels around the detected objects 
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
		key = cv2.waitKey(1)
		if key == 27:
			break
	cap.release()


'''
Detection using video
'''
def start_video(video_path):
    # load the pre-trained weights of the model YOLO-v3
	model, classes, colors, output_layers = load_yolo()
	cap = cv2.VideoCapture(video_path)
	while True:
        # generate individual frames of the video for detection
		_, frame = cap.read()
		height, width, channels = frame.shape
		blob, outputs = detect_objects(frame, model, output_layers)
        # detection of the images using 
		boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        # draw the labels around the boxes of the detected part of the video
		draw_labels(boxes, confs, colors, class_ids, classes, frame)

		key = cv2.waitKey(1)
		if cv2.waitKey(1) & 0xFF ==ord('q'):
			break
	cap.release()

if __name__ == '__main__':
	webcam = args.webcam
	video_play = args.play_video
	image = args.image
	if webcam:
		if args.verbose:
			print('---- Starting Web Cam object detection ----')
		webcam_detect()
	if video_play:
		video_path = args.video_path
		if args.verbose:
			print('Opening '+video_path+" .... ")
		start_video(video_path)
	if image:
		image_path = args.image_path
		if args.verbose:
			print("Opening "+image_path+" .... ")
		image_detect(image_path)
	

	cv2.destroyAllWindows()