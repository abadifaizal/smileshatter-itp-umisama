import cv2

def detect(imagePath, cascadePath):

	ESC_KEY = 27
	FRAME_RATE = 30
	INTERVAL = 1000 / FRAME_RATE

	ORG_WINDOW_NAME = "Original"

	DEVICE_ID = 0

	white = (0, 0, 255)



	cv2.namedWindow(ORG_WINDOW_NAME)

	cascade = cv2.CascadeClassifier(cascadePath)


	image = cv2.imread(imagePath)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	height, width, channels = image.shape
	face_list = cascade.detectMultiScale(image_gray, minSize=(100, 100))

	for (x,y,w,h) in face_list:
		color = (0, 0, 255)
		pen_w = 3
		cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness = pen_w)

	cv2.imshow(ORG_WINDOW_NAME, image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


	# cv2.destroyAllWindow()
	# cap.release()


detect("../images/cat_sample.jpg", "../cascades/cascade.xml")