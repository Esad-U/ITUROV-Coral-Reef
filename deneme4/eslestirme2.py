import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("2.2f.png")
img2 = cv2.imread("1.1f.png")

copyimg1 = img1.copy()
copyimg2 = img2.copy()

img1zero = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1k = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1y = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1m = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)

img2k = np.zeros((img2.shape[0],img2.shape[1],3), np.uint8)
img2y = np.zeros((img2.shape[0],img2.shape[1],3), np.uint8)
img2m = np.zeros((img2.shape[0],img2.shape[1],3), np.uint8)
img2zero = np.zeros((img2.shape[0],img2.shape[1],3), np.uint8)

kernel = np.ones((2,2),np.uint8)

def bitwise_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    bitwise1 = cv2.bitwise_and(img, img, mask=mask1)


    lower_beyaz = np.array([63, 28, 183])
    upper_beyaz = np.array([81, 255, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    bitwise2 = cv2.bitwise_and(img, img, mask=mask2)

    top_bitwise =cv2.add(bitwise1,bitwise2)
    return top_bitwise

def mask_alma_beyaz(img):
    lower_beyaz = np.array([63, 28, 183])
    upper_beyaz = np.array([81, 255, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    return mask2

def mask_alma_sari(img):
    lower_sari = np.array([20, 0, 0])
    upper_sari = np.array([50, 255, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_sari, upper_sari)
    return mask2

def mask_alma_pembe(img):
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1


pe1 = mask_alma_pembe(img1)
pe2 = mask_alma_pembe(img2)


dilation1 = cv2.dilate(pe1,kernel, iterations = 5)
dilation2 = cv2.dilate(pe2,kernel, iterations = 5)

erode1 = cv2.erode(pe1,kernel,iterations = 3)
erode2 = cv2.erode(pe2,kernel,iterations = 3)

cv2.imshow("Maske Pe 1",pe1)
#cv2.imshow("Maske Pe 2",pe2)

#cv2.imshow("Dilation 1",dilation1)
#cv2.imshow("Dilation 2",dilation2)

#cv2.imshow("Erode 1",erode1)
#cv2.imshow("Erode 2",erode2)

#######################################################################################################################

sobelSol = cv2.Sobel(erode1, cv2.CV_8U, 1, 0, ksize=1)
cv2.imshow("x8u 1", sobelSol)

sobelx = cv2.Sobel(erode1, cv2.CV_64F, 1, 0, ksize=1)
abs_sobelx = np.absolute(sobelx)
sobelX = np.uint8(abs_sobelx)

sobelSag = sobelX - sobelSol
cv2.imshow("sag", sobelSag)

contours, hierarchy = cv2.findContours(sobelSol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area = cv2.contourArea(contours[i])
    # print(area)
    if area < 30:
        # print("Çıkarıldı")
        pass
    else:
        # print("Eklendi")
        liste.insert(a, cv2.boundingRect(contours[i]))
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero, (x, y), (x + w, y + h), (0, 0, 255), -1)
        cv2.circle(img1k, (x, y), 25, (0, 0, 255), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a + 1

    # for i in range(len(contours)):
    # cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
    # for i in range(len(hull)):
    #    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

contours, hierarchy = cv2.findContours(sobelSag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area = cv2.contourArea(contours[i])
    # print(area)
    if area < 30:
        # print("Çıkarıldı")
        pass
    else:
        # print("Eklendi")
        liste.insert(a, cv2.boundingRect(contours[i]))
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.circle(img1y, (x, y), 25, (0, 255, 0), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a + 1

# for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
# for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)
img1ky = img1k + img1y

img1ky = mask_alma_sari(img1ky)
cv2.imshow("img1ky", img1ky)

contours, hierarchy = cv2.findContours(img1ky, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points1 = []

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    points1.append((int(x + (w / 2)),int(y + (h / 2))))
    cv2.circle(img1, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), 1, -1)

#######################################################################################################################


sobelSol = cv2.Sobel(erode2, cv2.CV_8U, 1, 0, ksize=1)
cv2.imshow("x8u 2", sobelSol)

sobelx = cv2.Sobel(erode2, cv2.CV_64F, 1, 0, ksize=1)
abs_sobelx = np.absolute(sobelx)
sobelX = np.uint8(abs_sobelx)

sobelSag = sobelX - sobelSol
cv2.imshow("sag", sobelSag)

contours, hierarchy = cv2.findContours(sobelSol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area = cv2.contourArea(contours[i])
    # print(area)
    if area < 30:
        # print("Çıkarıldı")
        pass
    else:
        # print("Eklendi")
        liste.insert(a, cv2.boundingRect(contours[i]))
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero, (x, y), (x + w, y + h), (0, 0, 255), -1)
        cv2.circle(img2k, (x, y), 25, (0, 0, 255), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a + 1

    # for i in range(len(contours)):
    # cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
    # for i in range(len(hull)):
    #    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

contours, hierarchy = cv2.findContours(sobelSag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area = cv2.contourArea(contours[i])
    # print(area)
    if area < 30:
        # print("Çıkarıldı")
        pass
    else:
        # print("Eklendi")
        liste.insert(a, cv2.boundingRect(contours[i]))
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero, (x, y), (x + w, y + h), (0, 255, 0), -1)
        cv2.circle(img2y, (x, y), 25, (0, 255, 0), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a + 1

# for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
# for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)
img2ky = img2k + img2y

img2ky = mask_alma_sari(img2ky)
cv2.imshow("img1ky", img2ky)

contours, hierarchy = cv2.findContours(img2ky, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

points2 = []

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    points2.append((int(x + (w / 2)), int(y + (h / 2))))
    cv2.circle(img2, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), 1, -1)

print(points1)
print(points2)

keypoints1 = np.float32(points1)
keypoints2 = np.float32(points2)



h, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC)
height, width, channels = img2.shape
warped = cv2.warpPerspective(copyimg1, h, (width, height))

added = cv2.addWeighted(bitwise_alma(warped),0.5,copyimg2,0.5,0)
cv2.imshow("Added", added)


#cv2.imshow("Mavi",img1m)
#cv2.imshow("Yeşil",img1y)
#cv2.imshow("Kırmızı",img1k)


cv2.imshow("img1zero",img1zero)
#cv2.imshow("Erode 2",erode2)

cv2.imshow("IMG 1",img1)
cv2.imshow("IMG 2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()