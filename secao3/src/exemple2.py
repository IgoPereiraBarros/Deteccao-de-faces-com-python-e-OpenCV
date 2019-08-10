import cv2


classifier_face = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
classifier_eye = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

image = cv2.imread('pessoas/pessoas4.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = classifier_face.detectMultiScale(image_gray)

for (x, y, w, h) in detected_faces:
	image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

	region = image[y: y + h, x: x + w]
	region_eye_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
	
	detected_eyes = classifier_eye.detectMultiScale(region_eye_gray, scaleFactor=1.08, minNeighbors=5, minSize=(5, 5))
	
	for (eye_x, eye_y, eye_w, eye_h) in detected_eyes:
		cv2.rectangle(region, (eye_x, eye_y), (eye_x+eye_w, eye_y+eye_h), (0, 255, 0), 2)

cv2.imshow('Faces e olhos detectados', image)
cv2.waitKey()

