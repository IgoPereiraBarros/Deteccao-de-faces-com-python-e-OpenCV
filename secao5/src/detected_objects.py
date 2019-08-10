import cv2

#classifier_cat = cv2.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')
#classifier_clock = cv2.CascadeClassifier('cascades/relogios.xml')
#classifier_car = cv2.CascadeClassifier('cascades/cars.xml')
classifier_banana = cv2.CascadeClassifier('../banana-classifier.xml')

image = cv2.imread('../banana1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detected = classifier_cat.detectMultiScale(image_gray, scaleFactor=1.02)
#detected = classifier_clock.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1)
#detected = classifier_car.detectMultiScale(image_gray, scaleFactor=1.01)
detected = classifier_banana.detectMultiScale(image_gray)
print(detected)

for (x, y, w, h) in detected:
    image = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    '''if w > 50 or h > 50:
                	print('Foi detectado uma anomalia na detecção dos carros, fora do padrão')'''

cv2.imshow("Gatos encontrados", image)
cv2.waitKey()

