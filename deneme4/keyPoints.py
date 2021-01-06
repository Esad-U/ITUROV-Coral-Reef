import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("1.1f.png")
img2 = cv2.imread("1.1f.png")

copyimg1 = img1.copy()
copyimg2 = img2.copy()

kernel = np.ones((2,2),np.uint8)


def bitwise_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    bitwise1 = cv2.bitwise_and(img, img, mask=mask1)


    lower_beyaz = np.array([90, 14, 178])
    upper_beyaz = np.array([137, 68, 255])

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

def mask_alma_pembe(img):
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

def nokta_bulma(liste):
    yListesi = []
    xListesi = []
    indislerSol = []
    indislerSag = []
    sonListe = []
    solAyak = []
    sagAyak = []
    a = 0
    while a < 4:
        for i in range(len(liste)):
            (x,y,w,h) = liste[i]
            yListesi.append(y+h)
        print(yListesi)
        maxY = max(yListesi)
        indis = yListesi.index(maxY)
        yListesi = []
        m = liste[indis]
        sonListe.append(m)
        del liste[indis]
        a += 1

    a = 0
    while a < 4:
        (x, y, w, h) = sonListe[a]
        xListesi.append(x)
        a += 1
    copyXListe = xListesi.copy()
    maxX1 = max(xListesi)
    indis4 = xListesi.index(maxX1)
    minX1 = min(xListesi)
    indis1 = xListesi.index(minX1)
    copyXListe.remove(maxX1)
    copyXListe.remove(minX1)
    maxX2 =max(copyXListe)
    indis3 = xListesi.index(maxX2)
    minX2 = min(copyXListe)
    indis2 = xListesi.index(minX2)
    print(xListesi)
    print(indis1)
    print(indis2)
    print(indis3)
    print(indis4)

    solAyak.append(sonListe[indis1])
    solAyak.append(sonListe[indis2])
    sagAyak.append(sonListe[indis3])
    sagAyak.append(sonListe[indis4])
    print(solAyak)
    print(sagAyak)

    ustSayısıSol = 0
    for i in range(len(liste)):
        (x, y, w, h) = liste[i]
        if x > minX1-15 and x < minX2+10:
            ustSayısıSol += 1
            indislerSol.append(i)

    print("Üst Sayısı Sol = " + str(ustSayısıSol))
    print(indislerSol)

    ustSayısıSag = 0
    for i in range(len(liste)):
        (x, y, w, h) = liste[i]
        if x > maxX2 - 15 and x < maxX1 + 10:
            ustSayısıSag += 1
            indislerSag.append(i)

    print("Üst Sayısı Sag = " + str(ustSayısıSag))
    print(indislerSag)

    return (sonListe, solAyak, sagAyak, indislerSag, indislerSol,indis1,indis2)

pe1 = mask_alma_pembe(img1)
pe2 = mask_alma_pembe(img2)

pe1 = cv2.erode(pe1,kernel,iterations = 5)
pe2 = cv2.erode(pe2,kernel,iterations = 5)

###################################################################

img = pe1
# Output dtype = cv.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
opening = cv2.morphologyEx(sobel_8u, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
opening3 = cv2.morphologyEx(opening2, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

fark = np.zeros((opening3.shape[0], opening3.shape[1], 3), np.uint8)

liste1 = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area < 70:
        #print("Çıkarıldı")
        pass
    else:
        #print("Eklendi")
        liste1.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


for i in range(len(contours)):
    cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
for i in range(len(hull)):
    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

###################################################################

img = pe2
# Output dtype = cv.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
opening = cv2.morphologyEx(sobel_8u, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
opening3 = cv2.morphologyEx(opening2, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

fark = np.zeros((opening3.shape[0], opening3.shape[1], 3), np.uint8)

liste2 = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area < 70:
        #print("Çıkarıldı")
        pass
    else:
        #print("Eklendi")
        liste2.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


for i in range(len(contours)):
    cv2.drawContours(img2, contours, i, (255, 255, 255), 1, 8)
for i in range(len(hull)):
    cv2.drawContours(img2, hull, i, (0, 255, 0), 1, 8)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

###################################################################


sonListe1, solAyak1, sagAyak1, indislerSag1, indislerSol1, indis11, indis12 = nokta_bulma(liste1)
sonListe2, solAyak2, sagAyak2, indislerSag2, indislerSol2, indis21, indis22 = nokta_bulma(liste2)

#print(sonListe1)

###################################################################

a = 0
while a < 4:
    (x,y,w,h) = sonListe1[a]
    cv2.circle(img1, (x + w, y + h), 5, (255, 0, 0), 1, -1)
    cv2.circle(img1, (x, y + h), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSol1):
    k = indislerSol1[a]
    (x,y,w,h) = liste1[k]
    cv2.circle(img1, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSag1):
    k = indislerSag1[a]
    (x,y,w,h) = liste1[k]
    cv2.circle(img1, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1

###################################################################

a = 0
while a < 4:
    (x,y,w,h) = sonListe2[a]
    cv2.circle(img2, (x + w, y + h), 5, (255, 0, 0), 1, -1)
    cv2.circle(img2, (x, y + h), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSol2):
    k = indislerSol2[a]
    (x,y,w,h) = liste2[k]
    cv2.circle(img2, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSag2):
    k = indislerSag2[a]
    (x,y,w,h) = liste2[k]
    cv2.circle(img2, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1

###################################################################
nokta11= sonListe1[indis11]
nokta12= sonListe1[indis12]

###################################################################

solNoktalar1 = []
a = 0
while a < len(indislerSol1):
    k = indislerSol1[a]
    (x, y, w, h) = liste1[k]
    solNoktalar1.append(y)
    a += 1
solUstNoktaY1 = max(solNoktalar1)
solUstNoktaIndis11 = solNoktalar1.index(solUstNoktaY1)
solUstNoktaIndis12 = indislerSol1[solUstNoktaIndis11]
solUstNoktaKor1 = liste1[solUstNoktaIndis12]
(x,y,w,h) = solUstNoktaKor1
cv2.circle(img1, (x, y), 8, (125, 0, 125), 2, -1)


sagNoktalar1 = []
a = 0
while a < len(indislerSag1):
    k = indislerSag1[a]
    (x, y, w, h) = liste1[k]
    sagNoktalar1.append(y)
    a += 1
sagUstNoktaY1 = max(sagNoktalar1)
sagUstNoktaIndis11 = sagNoktalar1.index(sagUstNoktaY1)
sagUstNoktaIndis12 = indislerSag1[sagUstNoktaIndis11]
sagUstNoktaKor1 = liste1[sagUstNoktaIndis12]
(x,y,w,h) = sagUstNoktaKor1
cv2.circle(img1, (x, y), 8, (125, 0, 125), 2, -1)

###################################################################

solNoktalar2 = []
a = 0
while a < len(indislerSol2):
    k = indislerSol2[a]
    (x, y, w, h) = liste2[k]
    solNoktalar2.append(y)
    a += 1
solUstNoktaY2 = max(solNoktalar2)
solUstNoktaIndis21 = solNoktalar2.index(solUstNoktaY2)
solUstNoktaIndis22 = indislerSol2[solUstNoktaIndis21]
solUstNoktaKor2 = liste2[solUstNoktaIndis22]
(x,y,w,h) = solUstNoktaKor2
cv2.circle(img2, (x, y), 8, (125, 0, 125), 2, -1)


sagNoktalar2 = []
a = 0
while a < len(indislerSag2):
    k = indislerSag2[a]
    (x, y, w, h) = liste2[k]
    sagNoktalar2.append(y)
    a += 1
sagUstNoktaY2 = max(sagNoktalar2)
sagUstNoktaIndis21 = sagNoktalar2.index(sagUstNoktaY2)
sagUstNoktaIndis22 = indislerSag2[sagUstNoktaIndis21]
sagUstNoktaKor2 = liste2[sagUstNoktaIndis22]
(x,y,w,h) = sagUstNoktaKor2
cv2.circle(img2, (x, y), 8, (125, 0, 125), 2, -1)

###################################################################

solNoktalar1 = []
a = 0

while a < len(solAyak1):
    (x, y, w, h) = solAyak1[a]
    solNoktalar1.append(x)
    a += 1
solAltNoktaX1 = min(solNoktalar1)
solAltNoktaIndis1 = solNoktalar1.index(solAltNoktaX1)
solAltNoktaKor1 = solAyak1[solAltNoktaIndis1]
(x,y,w,h) = solAltNoktaKor1
cv2.circle(img1, (x, y + h), 8, (125, 0, 125), 2, -1)

sagNoktalar1 = []
a = 0

while a < len(sagAyak1):
    (x, y, w, h) = sagAyak1[a]
    sagNoktalar1.append(x)
    a += 1
sagAltNoktaX1 = max(sagNoktalar1)
sagAltNoktaIndis1 = sagNoktalar1.index(sagAltNoktaX1)
sagAltNoktaKor1 = sagAyak1[sagAltNoktaIndis1]
(x,y,w,h) = sagAltNoktaKor1
cv2.circle(img1, (x + w, y + h), 8, (125, 0, 125), 2, -1)

###################################################################

solNoktalar2 = []
a = 0

while a < len(solAyak2):
    (x, y, w, h) = solAyak2[a]
    solNoktalar2.append(x)
    a += 1
solAltNoktaX2 = min(solNoktalar2)
solAltNoktaIndis2 = solNoktalar2.index(solAltNoktaX2)
solAltNoktaKor2 = solAyak2[solAltNoktaIndis2]
(x,y,w,h) = solAltNoktaKor2
cv2.circle(img2, (x, y + h), 8, (125, 0, 125), 2, -1)

sagNoktalar2 = []
a = 0

while a < len(sagAyak2):
    (x, y, w, h) = sagAyak2[a]
    sagNoktalar2.append(x)
    a += 1
sagAltNoktaX2 = max(sagNoktalar2)
sagAltNoktaIndis2 = sagNoktalar2.index(sagAltNoktaX2)
sagAltNoktaKor2 = sagAyak2[sagAltNoktaIndis2]
(x,y,w,h) = sagAltNoktaKor2
cv2.circle(img2, (x + w, y + h), 8, (125, 0, 125), 2, -1)

###################################################################
###################################################################

(x,y,w,h) = solUstNoktaKor1
pts1 = [x, y]
(x,y,w,h) = sagUstNoktaKor1
pts2 = [x, y]
(x,y,w,h) = solAltNoktaKor1
pts3 = [x, y + h]
(x,y,w,h) = sagAltNoktaKor1
pts4 = [x + w, y + h]

points1 = np.float32([pts1, pts2, pts3, pts4])


(x,y,w,h) = solUstNoktaKor2
pts1 = [x, y]
(x,y,w,h) = sagUstNoktaKor2
pts2 = [x, y]
(x,y,w,h) = solAltNoktaKor2
pts3 = [x, y + h]
(x,y,w,h) = sagAltNoktaKor2
pts4 = [x + w, y + h]

points2 = np.float32([pts1, pts2, pts3, pts4])

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
height, width, channels = img2.shape
warped = cv2.warpPerspective(copyimg1, h, (width, height))

added = cv2.addWeighted(bitwise_alma(warped),0.5,copyimg2,0.5,0)
cv2.imshow("Added", added)

#opening3 = cv2.resize(opening3,(opening3.shape[1]*2,opening3.shape[0]*2))
#img1 = cv2.resize(img1,(img1.shape[1]*2,img1.shape[0]*2))
#cv2.imshow("Opening3",opening3)
cv2.imshow("Reef1",img1)
cv2.imshow("Reef2",img2)
cv2.imshow("Warped",warped)
cv2.waitKey(0)
cv2.destroyAllWindows()