import cv2 as cv


face_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
img=cv.imread("onstage_2.png")

img_res=cv.resize(img, (500,500))
gray_img = cv.cvtColor(img_res, cv.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(0,0))
for (x, y, w, h) in faces:
    cv.rectangle(img_res, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv.imshow("image 1", img_res)
cv.waitKey(0)
cv.destroyAllWindows()