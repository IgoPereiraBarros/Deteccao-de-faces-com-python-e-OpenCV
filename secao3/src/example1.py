import cv2

cascade_classifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

image = cv2.imread('pessoas/pessoas1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#faces_detected = cascade_classifier.detectMultiScale(image_grey, scaleFactor=1.08, minNeighbors=9, minSize=(30, 30))
faces_detected = cascade_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minSize=(30, 30), minNeighbors=3)

print(len(faces_detected))
print()
print(faces_detected)
print()

'''
	x --> eixo X
	y --> eixo Y
	w --> largura
	h --> altura

'''
for  (x, y, w, h) in faces_detected:
	#print(x, y, w, h)
	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2) # rectangle(imagem, posição da imagem em relação aos eixos X e Y, o formato do retângulo ao redor do rosto, a cor do retangulo, e a largura do retangulo)

cv2.imshow('Faces detected', image)
cv2.waitKey()