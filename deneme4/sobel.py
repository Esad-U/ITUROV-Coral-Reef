import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("1.1f.png")
img2 = cv2.imread("1.1f.png")

img1zero = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1k = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1y = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
img1m = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
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


dilation1 = cv2.dilate(pe1,kernel, iterations = 5)
dilation2 = cv2.dilate(pe2,kernel, iterations = 5)

erode1 = cv2.erode(pe1,kernel,iterations = 3)
erode2 = cv2.erode(pe2,kernel,iterations = 3)



points =[]

sobelSol = cv2.Sobel(erode1,cv2.CV_8U,1,0,ksize=1)
#cv2.imshow("x8u 1",sobelSol)

sobelx = cv2.Sobel(erode1,cv2.CV_64F,1,0,ksize=1)
abs_sobelx = np.absolute(sobelx)
sobelX = np.uint8(abs_sobelx)

sobelSag = sobelX-sobelSol
#cv2.imshow("sag",sobelSag)

contours, hierarchy = cv2.findContours(sobelSol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area > 50:
        #print("Eklendi")
        liste.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x+3,y),(x+w+3,y+h),(0,0,255),-1)
        cv2.circle(img1k, (x+3, y), 25, (0, 0, 255), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


#for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
#for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

contours, hierarchy = cv2.findContours(sobelSag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area > 30:
        #print("Eklendi")
        liste.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x-3,y),(x+w-3,y+h),(0,255,0),-1)
        cv2.circle(img1y, (x-3, y), 25, (0, 255, 0), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


#for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
#for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

sobelUst = cv2.Sobel(erode1,cv2.CV_8U,0,1,ksize=1)
#cv2.imshow("Ust",sobelUst)

sobely = cv2.Sobel(erode1,cv2.CV_64F,0,1,ksize=1)
abs_sobely = np.absolute(sobely)
sobelY = np.uint8(abs_sobely)

sobelAlt = sobelY - sobelUst
#cv2.imshow("Alt",sobelAlt)

contours, hierarchy = cv2.findContours(sobelUst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []

liste = []
for i in range(len(contours)):
    a = 0
    area =cv2.contourArea(contours[i])
    #print(area)
    if area > 40:
        #print("Eklendi")
        liste.insert(a,cv2.boundingRect(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x,y),(x+w,y+h),(255,0,0),-1)
        cv2.circle(img1m, (x, y), 5, (255, 0, 0), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))
        a = a+1


#for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
#for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

contours, hierarchy = cv2.findContours(sobelAlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

hull = []
listeAlt = []
listeAltAlan = []

sonArea = 10000

for i in range(len(contours)):
    area =cv2.contourArea(contours[i])
    if area > 10:
        listeAlt.append(cv2.boundingRect(contours[i]))
        listeAltAlan.append(cv2.contourArea(contours[i]))
        (x,y,w,h) = cv2.boundingRect(contours[i])
        cv2.rectangle(img1zero,(x,y),(x+w,y+h),(255,255,0),1)
        cv2.circle(img1m, (x, y), 5, (255, 255, 0), -1, -1)
        hull.append(cv2.convexHull(contours[i], False))

sortedListeAltAlan = sorted(listeAltAlan)
if sortedListeAltAlan[0] == sortedListeAltAlan[1]:
    g = (i for i, n in enumerate(listeAltAlan) if n == sortedListeAltAlan[0])
    kucuk1index = next(g)
    kucuk2index = next(g)
else :
    kucuk1index = listeAltAlan.index(sortedListeAltAlan[0])
    kucuk2index = listeAltAlan.index(sortedListeAltAlan[1])


print(sortedListeAltAlan)
print(listeAltAlan)
print(kucuk1index)
print(kucuk2index)

(x1,y1,w1,h1) = listeAlt[kucuk1index]
cv2.circle(img1, (int(x1+(w1)/2), int(y1+(h1)/2)), 3, (255, 0, 255), -1, -1)
(x2,y2,w2,h2) = listeAlt[kucuk2index]
cv2.circle(img1, (int(x2+(w2)/2), int(y2+(h2)/2)), 3, (255, 0, 255), -1, -1)

if x1 > x2:
    points.append((int(x2+(w2)/2), int(y2+(h2)/2)))
    points.append((int(x1+(w1)/2), int(y1+(h1)/2)))
else:
    points.append((int(x1+(w1)/2), int(y1+(h1)/2)))
    points.append((int(x2+(w2)/2), int(y2+(h2)/2)))

#for i in range(len(contours)):
# cv2.drawContours(img1, contours, i, (255, 255, 255), 1, 8)
#for i in range(len(hull)):
#    cv2.drawContours(img1, hull, i, (0, 255, 0), 1, 8)

cv2.imshow("Mavi",img1m)
cv2.imshow("Yeşil",img1y)
cv2.imshow("Kırmızı",img1k)

img1ky = img1k + img1y
cv2.imshow("img1ky",img1ky)
sariMask = mask_alma_sari(img1ky)
cv2.imshow("Sari",sariMask)

contours, hierarchy = cv2.findContours(sariMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    cv2.circle(img1, (int(x+(w/2)), int(y+(h/2))), 5, (255, 0, 0), 1, -1)
    cv2.circle(img1ky, (int(x + (w / 2)), int(y + (h / 2))), 50, (0, 0, 0), -1, -1)
cv2.imshow("son",img1ky)

imgKS = mask_alma_kirmizi_yesil(img1ky)
contours, hierarchy = cv2.findContours(imgKS, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    (x, y, w, h) = cv2.boundingRect(contours[i])
    points.append((int(x+(w/2)), int(y+(h/2))))
    cv2.circle(img1, (int(x+(w/2)), int(y+(h/2))), 5, (0, 0, 255), 1, -1)

print(points)

#cv2.imshow("sobel X",sobelX)
cv2.imshow("img1zero",img1zero)
#cv2.imshow("Erode 2",erode2)

cv2.imshow("IMG 1",img1)
#cv2.imshow("IMG 2",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()