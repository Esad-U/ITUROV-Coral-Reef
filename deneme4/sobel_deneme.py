import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("2.2f.png")
img2 = cv2.imread("2.1f.png")

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
    lower_beyaz = np.array([90, 14, 178])
    upper_beyaz = np.array([137, 68, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    return mask2

def mask_alma_pembe(img):
    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel




#cv2.imshow("Bitwise 1",bitwise_alma(img1))
#cv2.imshow("Bitwise 2",bitwise_alma(img2))

be1 =mask_alma_beyaz(img1)
be2 =mask_alma_beyaz(img2)

eroBe1 = cv2.erode(be1,kernel,iterations = 3)
eroBe2 = cv2.erode(be2,kernel,iterations = 3)

#cv2.imshow("Maske Be 1",be1)
#cv2.imshow("Maske Be 2",be2)
#cv2.imshow("Erosion Be 1",eroBe1)
#cv2.imshow("Erosion Be 2",eroBe2)

pe1 = mask_alma_pembe(img1)
pe2 = mask_alma_pembe(img2)

cv2.imshow("Maske Pe 1",pe1)
cv2.imshow("Maske Pe 2",pe2)

pe1 = cv2.erode(pe1,kernel,iterations = 3)
pe2 = cv2.erode(pe2,kernel,iterations = 3)

#cv2.imshow("skeleton1",skeletonize(pe1))
#cv2.imshow("skeleton2",skeletonize(pe2))

img = cv2.dilate(pe1,kernel,iterations= 3)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# Output dtype = cv.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
cv2.imshow("sol",sobelx8u)
# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)
#sobel_8u = cv2.medianBlur(sobel_8u,1)
#sobel_8u = cv2.blur(sobel_8u,(3,3))

opening = cv2.morphologyEx(sobel_8u, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
opening3 = cv2.morphologyEx(opening2, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

hull = []

fark = np.zeros((opening3.shape[0], opening3.shape[1], 3), np.uint8)

liste = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area < 30:
        #print("Çıkarıldı")
        pass
    else:
        #print("Eklendi")
        liste.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


for i in range(len(contours)):
    cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
for i in range(len(hull)):
    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

cv2.imshow("Opening1",opening)
#cv2.imshow("Opening2",opening2)
#cv2.imshow("Opening3",opening3)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()


def nokta_bulma(liste):
    yListesi = []
    xListesi = []
    indislerSol = []
    indislerSag = []
    sonListe = []
    copyXListe = []
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
        if x > minX1-10 and x < minX2+10:
            ustSayısıSol += 1
            indislerSol.append(i)

    print("Üst Sayısı Sol = " + str(ustSayısıSol))
    print(indislerSol)

    ustSayısıSag = 0
    for i in range(len(liste)):
        (x, y, w, h) = liste[i]
        if x > maxX2 - 10 and x < maxX1 + 10:
            ustSayısıSag += 1
            indislerSag.append(i)

    print("Üst Sayısı Sag = " + str(ustSayısıSag))
    print(indislerSag)

    return (sonListe, solAyak, sagAyak, indislerSag, indislerSol)

sonListe, solAyak, sagAyak, indislerSag, indislerSol = nokta_bulma(liste)

print(sonListe)

a = 0
while a < 4:
    (x,y,w,h) = sonListe[a]
    cv2.circle(img1, (x + w, y + h), 5, (255, 0, 0), 1, -1)
    cv2.circle(img1, (x, y + h), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSol):
    k = indislerSol[a]
    (x,y,w,h) = liste[k]
    cv2.circle(img1, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1


a = 0
while a < len(indislerSag):
    k = indislerSag[a]
    (x,y,w,h) = liste[k]
    cv2.circle(img1, (x, y), 5, (255, 0, 0), 1, -1)
    a += 1


#opening3 = cv2.resize(opening3,(opening3.shape[1]*2,opening3.shape[0]*2))
#img1 = cv2.resize(img1,(img1.shape[1]*2,img1.shape[0]*2))

cv2.imshow("Reef1",img1)
cv2.imshow("Reef2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()