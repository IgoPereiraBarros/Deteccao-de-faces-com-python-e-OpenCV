import cv2

image = cv2.imread('../opencv-python.jpg')
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', image)
cv2.imshow('Imagem Cinza', image_grey)
cv2.waitKey()
