import cv2
import numpy as np

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

    hsvSari = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskSari = cv2.inRange(hsvSari, lower_sari, upper_sari)
    return maskSari

def mask_alma_kirmizi_yesil(img):
    lower_renk = np.array([0, 255, 255])
    upper_renk = np.array([70, 255, 255])

    hsvKS = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskKS = cv2.inRange(hsvKS, lower_renk, upper_renk)
    return maskKS

def mask_alma_pembe(img):
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

pe1 = mask_alma_pembe(img1)
pe2 = mask_alma_pembe(img2)

erode1 = cv2.erode(pe1,kernel,iterations = 3)
erode2 = cv2.erode(pe2,kernel,iterations = 3)

points1 = []
points2 = []

#######################################################################################################################

sobelSol1 = cv2.Sobel(erode1,cv2.CV_8U,1,0,ksize=1)

sobelx1 = cv2.Sobel(erode1,cv2.CV_64F,1,0,ksize=1)
abs_sobelx1 = np.absolute(sobelx1)
sobelX1 = np.uint8(abs_sobelx1)

sobelSag1 = sobelX1-sobelSol1

contours, hierarchy = cv2.findContours(sobelSol1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    if area > 50:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x+3,y),(x+w+3,y+h),(0,0,255),-1)
        cv2.circle(img1k, (x+3, y), 25, (0, 0, 255), -1, -1)



contours, hierarchy = cv2.findContours(sobelSag1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 30:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x-3,y),(x+w-3,y+h),(0,255,0),-1)
        cv2.circle(img1y, (x-3, y), 25, (0, 255, 0), -1, -1)

sobelUst1 = cv2.Sobel(erode1,cv2.CV_8U,0,1,ksize=1)

sobely1 = cv2.Sobel(erode1,cv2.CV_64F,0,1,ksize=1)
abs_sobely1 = np.absolute(sobely1)
sobelY1 = np.uint8(abs_sobely1)

sobelAlt1 = sobelY1 - sobelUst1

contours, hierarchy = cv2.findContours(sobelUst1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 40:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x,y),(x+w,y+h),(255,0,0),-1)
        cv2.circle(img1m, (x, y), 5, (255, 0, 0), -1, -1)

contours, hierarchy = cv2.findContours(sobelAlt1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

listeAlt1 = []
listeAltAlan1 = []

for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 10:
        listeAlt1.append(cv2.boundingRect(contours[i]))
        listeAltAlan1.append(cv2.contourArea(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x,y),(x+w,y+h),(255,255,0),1)
        cv2.circle(img1m, (x, y), 5, (255, 255, 0), -1, -1)

sortedListeAltAlan = sorted(listeAltAlan1)
if sortedListeAltAlan[0] == sortedListeAltAlan[1]:
    g = (i for i, n in enumerate(listeAltAlan1) if n == sortedListeAltAlan[0])
    kucuk1index = next(g)
    kucuk2index = next(g)
else :
    kucuk1index = listeAltAlan1.index(sortedListeAltAlan[0])
    kucuk2index = listeAltAlan1.index(sortedListeAltAlan[1])


print(sortedListeAltAlan)
print(listeAltAlan1)
print(kucuk1index)
print(kucuk2index)

(x1,y1,w1,h1) = listeAlt1[kucuk1index]
cv2.circle(img1, (int(x1+(w1)/2), int(y1+(h1)/2)), 3, (255, 0, 255), -1, -1)
(x2,y2,w2,h2) = listeAlt1[kucuk2index]
cv2.circle(img1, (int(x2+(w2)/2), int(y2+(h2)/2)), 3, (255, 0, 255), -1, -1)

if x1 > x2:
    points1.append((int(x2+(w2)/2), int(y2+(h2)/2)))
    points1.append((int(x1+(w1)/2), int(y1+(h1)/2)))
else:
    points1.append((int(x1+(w1)/2), int(y1+(h1)/2)))
    points1.append((int(x2+(w2)/2), int(y2+(h2)/2)))


img1ky = img1k + img1y
sariMask = mask_alma_sari(img1ky)

contours, hierarchy = cv2.findContours(sariMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    cv2.circle(img1, (int(x+(w/2)), int(y+(h/2))), 5, (255, 0, 0), 1, -1)
    cv2.circle(img1ky, (int(x + (w / 2)), int(y + (h / 2))), 50, (0, 0, 0), -1, -1)
cv2.imshow("son",img1ky)

imgKS = mask_alma_kirmizi_yesil(img1ky)
contours, hierarchy = cv2.findContours(imgKS, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sutun1 = []
sutunX = []

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    sutun1.append((int(x+(w/2)), int(y+(h/2))))
    sutunX.append(x)
    cv2.circle(img1, (int(x+(w/2)), int(y+(h/2))), 5, (0, 0, 255), 1, -1)


for i in range(4):
    nokta1X = min(sutunX)
    nokta1index = sutunX.index(nokta1X)
    points1.append(sutun1[nokta1index])
    del sutunX[nokta1index]
    del sutun1[nokta1index]

print(points1)

#######################################################################################################################


sobelSol2 = cv2.Sobel(erode2,cv2.CV_8U,1,0,ksize=1)

sobelx2 = cv2.Sobel(erode2,cv2.CV_64F,1,0,ksize=1)
abs_sobelx2 = np.absolute(sobelx2)
sobelX2 = np.uint8(abs_sobelx2)

sobelSag2 = sobelX2-sobelSol2

contours, hierarchy = cv2.findContours(sobelSol2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    if area > 50:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero,(x+3,y),(x+w+3,y+h),(0,0,255),-1)
        cv2.circle(img2k, (x+3, y), 25, (0, 0, 255), -1, -1)



contours, hierarchy = cv2.findContours(sobelSag2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    #print(area)
    if area > 30:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero,(x-3,y),(x+w-3,y+h),(0,255,0),-1)
        cv2.circle(img2y, (x-3, y), 25, (0, 255, 0), -1, -1)

sobelUst2 = cv2.Sobel(erode2,cv2.CV_8U,0,1,ksize=1)

sobely2 = cv2.Sobel(erode2,cv2.CV_64F,0,1,ksize=1)
abs_sobely2 = np.absolute(sobely2)
sobelY2 = np.uint8(abs_sobely2)

sobelAlt2 = sobelY2 - sobelUst2

contours, hierarchy = cv2.findContours(sobelUst2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 40:
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero,(x,y),(x+w,y+h),(255,0,0),-1)
        cv2.circle(img2m, (x, y), 5, (255, 0, 0), -1, -1)

contours, hierarchy = cv2.findContours(sobelAlt2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

listeAlt2 = []
listeAltAlan2 = []

for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 10:
        listeAlt2.append(cv2.boundingRect(contours[i]))
        listeAltAlan2.append(cv2.contourArea(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2zero,(x,y),(x+w,y+h),(255,255,0),1)
        cv2.circle(img2m, (x, y), 5, (255, 255, 0), -1, -1)

sortedListeAltAlan2 = sorted(listeAltAlan2)
if sortedListeAltAlan2[0] == sortedListeAltAlan2[1]:
    g = (i for i, n in enumerate(listeAltAlan2) if n == sortedListeAltAlan2[0])
    kucuk21index = next(g)
    kucuk22index = next(g)
else :
    kucuk21index = listeAltAlan2.index(sortedListeAltAlan2[0])
    kucuk22index = listeAltAlan2.index(sortedListeAltAlan2[1])


print(sortedListeAltAlan2)
print(listeAltAlan2)
print(kucuk21index)
print(kucuk22index)

(x1,y1,w1,h1) = listeAlt2[kucuk21index]
cv2.circle(img2, (int(x1+(w1)/2), int(y1+(h1)/2)), 3, (255, 0, 255), -1, -1)
(x2,y2,w2,h2) = listeAlt2[kucuk22index]
cv2.circle(img2, (int(x2+(w2)/2), int(y2+(h2)/2)), 3, (255, 0, 255), -1, -1)

if x1 > x2:
    points2.append((int(x2+(w2)/2), int(y2+(h2)/2)))
    points2.append((int(x1+(w1)/2), int(y1+(h1)/2)))
else:
    points2.append((int(x1+(w1)/2), int(y1+(h1)/2)))
    points2.append((int(x2+(w2)/2), int(y2+(h2)/2)))

img2ky = img2k + img2y
sariMask2 = mask_alma_sari(img2ky)

contours, hierarchy = cv2.findContours(sariMask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    cv2.circle(img2, (int(x+(w/2)), int(y+(h/2))), 5, (255, 0, 0), 1, -1)
    cv2.circle(img2ky, (int(x + (w / 2)), int(y + (h / 2))), 50, (0, 0, 0), -1, -1)
cv2.imshow("son 2",img2ky)

imgKS2 = mask_alma_kirmizi_yesil(img2ky)
contours, hierarchy = cv2.findContours(imgKS2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sutun1 = []
sutunX = []

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    sutun1.append((int(x + (w / 2)), int(y + (h / 2))))
    sutunX.append(x)
    cv2.circle(img2, (int(x+(w/2)), int(y+(h/2))), 5, (0, 0, 255), 1, -1)


for i in range(4):
    nokta1X = min(sutunX)
    nokta1index = sutunX.index(nokta1X)
    points2.append(sutun1[nokta1index])
    del sutunX[nokta1index]
    del sutun1[nokta1index]

print(points2)

keypoints1 = np.float32(points1)
keypoints2 = np.float32(points2)

h, mask = cv2.findHomography(keypoints1, keypoints2, cv2.RANSAC)
height, width, channels = img2.shape
warped = cv2.warpPerspective(copyimg1, h, (width, height))

added = cv2.addWeighted(bitwise_alma(warped),0.5,copyimg2,0.5,0)
cv2.imshow("Added", added)


cv2.imshow("img1zero",img1zero)
cv2.imshow("img2zero",img2zero)

cv2.imshow("IMG 1",img1)
cv2.imshow("IMG 2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()