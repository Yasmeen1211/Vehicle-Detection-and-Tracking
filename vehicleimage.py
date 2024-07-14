import cv2
import imutils

cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
image_path = 'nocar.jpg'  
img = cv2.imread(image_path)

# Print the image variable to check if it's None
print("Image:", img)

img = imutils.resize(img, width=800)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# Draw rectangles around the detected vehicles
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the image with the detected vehicles
cv2.imshow("Detected Vehicles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
