import cv2


video = cv2.VideoCapture(0)
classifier_faces = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

while True:
	conected, frame = video.read()

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detected_faces = classifier_faces.detectMultiScale(frame_gray, minSize=(70, 70))

	for (x, y, w, h) in detected_faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('Vídeo', frame)

	if cv2.waitKey(1) == ord('c'):
		break

video.release()
cv2.destroyAllWindows()