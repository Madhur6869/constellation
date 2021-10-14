import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

color_bgr=cv2.imread('./Stars/noise/draco_noise.jpg')


gray=cv2.cvtColor(color_bgr,cv2.COLOR_BGR2GRAY)

# show the plotting graph of an image
# histr = cv2.calcHist([gray], [0], None, [256], [0, 256])
# plt.plot(histr)
# plt.show()

cv2.imshow('Gray image',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Convert pixels < 190 to 0 and others to 255

# ret,binary=cv2.threshold(gray,190,255,cv2.THRESH_OTSU)
ret,binary=cv2.threshold(gray,190,255,cv2.THRESH_BINARY_INV)
cv2.imshow('Binary image',binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

inverted_binary=~binary
cv2.imshow('Inverted binary image',inverted_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Finding all contours i.e pixels with similar intensity
contours, hierarchy=cv2.findContours(inverted_binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(color_bgr.shape[:2], dtype=color_bgr.dtype)
sum=0;
count=0
# draw all contours larger than avg on the mask
for c in contours:
    sum+=cv2.contourArea(c)
    count+=1
avg=sum/count
print(count,avg)
for c in contours:
    if cv2.contourArea(c) > avg/2:
        cv2.drawContours(mask, [c], 0, (255), -1)

# apply the mask to the original image
result = cv2.bitwise_and(color_bgr,color_bgr, mask= mask)

#show image
cv2.imshow("Result", result)

# img.imsave('Stars/templates/leo-2.jpg',result)

#Drawing around the contours; -1 is to draw all contours; color in BGR; Thickness
with_contours = cv2.drawContours(color_bgr, contours, -1, (255, 0, 255), 2)
# cv2.imshow('Detected contours', with_contours)
plt.imshow(with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('number of contours detected: ' + str(len(contours)))
