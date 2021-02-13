import cv2
import numpy as np

img1 = cv2.imread("2.2f.png")
img2 = cv2.imread("3.1.png")

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

def mask_alma(img):
    hsv1 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_pembe = np.array([148, 0, 100])
    upper_pembe = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)

    lower_beyaz = np.array([30, 0, 180])
    upper_beyaz = np.array([85, 100, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)

    top_mask =cv2.add(mask1,mask2)
    return top_mask

def mask_alma_beyaz(img):
    lower_beyaz = np.array([30, 0, 180])
    upper_beyaz = np.array([85, 100, 255])

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

def bit_xor(img1,img2,imgson):
    xor = cv2.bitwise_xor(img1, img2)
    #cv2.imshow("Xor", xor)
    xor_blur = cv2.medianBlur(xor,5)
    #cv2.imshow("Blur Xor",xor)

    contours, hierarchy = cv2.findContours(xor_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    fark = np.zeros((xor.shape[0], xor.shape[1], 3), np.uint8)

    for i in range(len(contours)):
        area =cv2.contourArea(contours[i])
        print(area)
        if area < 5000:
            print("Çıkarıldı")
        else:
            print("Eklendi")
            (x,y,w,h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgson,(x-5,y-5),(x+w+5,y+h+5),(0,0,255),1)
            hull.append(cv2.convexHull(contours[i], False))



    for i in range(len(contours)):
        cv2.drawContours(fark, contours, i, (255, 255, 255), 1, 8)
    for i in range(len(hull)):
        cv2.drawContours(fark, hull, i, (0, 255, 0), 1, 8)
    return (xor,xor_blur,fark)

def sobelFoto(gImg,img,imgk,imgy,imgm,imgzero):

    erode = cv2.erode(img, kernel, iterations=3)
    sobelSol = cv2.Sobel(erode, cv2.CV_8U, 1, 0, ksize=1)

    sobelx = cv2.Sobel(erode, cv2.CV_64F, 1, 0, ksize=1)
    abs_sobelx = np.absolute(sobelx)
    sobelX = np.uint8(abs_sobelx)

    sobelSag = sobelX - sobelSol

    listeSol = []

    contours, hierarchy = cv2.findContours(sobelSol, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for i in range(len(contours)):
        a = 0
        area = cv2.contourArea(contours[i])
        if area > 50:
            listeSol.append(cv2.boundingRect(contours[i]))
            (x, y, w, h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgzero, (x + 3, y), (x + w + 3, y + h), (0, 0, 255), -1)
            cv2.circle(imgk, (x + 3, y), 25, (0, 0, 255), -1, -1)

    listeSag = []

    contours, hierarchy = cv2.findContours(sobelSag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 30:
            listeSag.append(cv2.boundingRect(contours[i]))
            (x, y, w, h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgzero, (x - 3, y), (x + w - 3, y + h), (0, 255, 0), 3)
            cv2.circle(imgy, (x - 3, y), 25, (0, 255, 0), -1, -1)

    sobelUst = cv2.Sobel(erode, cv2.CV_8U, 0, 1, ksize=1)

    sobely = cv2.Sobel(erode, cv2.CV_64F, 0, 1, ksize=1)
    abs_sobely= np.absolute(sobely)
    sobelY = np.uint8(abs_sobely)

    sobelAlt = sobelY - sobelUst

    listeUst = []

    contours, hierarchy = cv2.findContours(sobelUst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 40:
            listeUst.append(cv2.boundingRect(contours[i]))
            (x, y, w, h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgzero, (x, y), (x + w, y + h), (255, 255, 0), -1)
            cv2.circle(imgm, (x, y), 5, (255, 0, 0), -1, -1)

    contours, hierarchy = cv2.findContours(sobelAlt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    listeAlt = []
    listeAltAlan = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 10:
            listeAlt.append(cv2.boundingRect(contours[i]))
            listeAltAlan.append(cv2.contourArea(contours[i]))
            (x, y, w, h) = cv2.boundingRect(contours[i])
            cv2.rectangle(imgzero, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.circle(imgm, (x, y), 5, (255, 255, 0), -1, -1)

    sortedListeAltAlan = sorted(listeAltAlan)
    if sortedListeAltAlan[0] == sortedListeAltAlan[1]:
        g = (i for i, n in enumerate(listeAltAlan) if n == sortedListeAltAlan[0])
        kucuk1index = next(g)
        kucuk2index = next(g)
    else:
        kucuk1index = listeAltAlan.index(sortedListeAltAlan[0])
        kucuk2index = listeAltAlan.index(sortedListeAltAlan[1])

    print(sortedListeAltAlan)
    print(listeAltAlan)
    print(kucuk1index)
    print(kucuk2index)

    (x1, y1, w1, h1) = listeAlt[kucuk1index]
    cv2.circle(gImg, (int(x1 + (w1) / 2), int(y1 + (h1) / 2)), 3, (255, 0, 255), -1, -1)
    (x2, y2, w2, h2) = listeAlt[kucuk2index]
    cv2.circle(gImg, (int(x2 + (w2) / 2), int(y2 + (h2) / 2)), 3, (255, 0, 255), -1, -1)

    points = []

    if x1 > x2:
        points.append((int(x2 + (w2) / 2), int(y2 + (h2) / 2)))
        points.append((int(x1 + (w1) / 2), int(y1 + (h1) / 2)))
    else:
        points.append((int(x1 + (w1) / 2), int(y1 + (h1) / 2)))
        points.append((int(x2 + (w2) / 2), int(y2 + (h2) / 2)))

    imgky = imgk + imgy
    sariMask = mask_alma_sari(imgky)

    contours, hierarchy = cv2.findContours(sariMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        cv2.circle(gImg, (int(x + (w / 2)), int(y + (h / 2))), 5, (255, 0, 0), 1, -1)
        cv2.circle(imgky, (int(x + (w / 2)), int(y + (h / 2))), 50, (0, 0, 0), -1, -1)

    imgKS = mask_alma_kirmizi_yesil(imgky)
    contours, hierarchy = cv2.findContours(imgKS, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sutun1 = []
    sutunX = []

    for i in range(len(contours)):
        (x, y, w, h) = cv2.boundingRect(contours[i])
        sutun1.append((int(x + (w / 2)), int(y + (h / 2))))
        sutunX.append(x)
        cv2.circle(gImg, (int(x + (w / 2)), int(y + (h / 2))), 5, (0, 0, 255), 1, -1)

    for i in range(4):
        nokta1X = min(sutunX)
        nokta1index = sutunX.index(nokta1X)
        points.append(sutun1[nokta1index])
        del sutunX[nokta1index]
        del sutun1[nokta1index]

    print(points)

    return (points,imgzero)

points1,cikan1 = sobelFoto(img1,mask_alma_pembe(img1),img1k,img1y,img1m,img1zero)
points2,cikan2 = sobelFoto(img2,mask_alma_pembe(img2),img2k,img2y,img2m,img2zero)

cv2.imshow("img1zero",cikan1)
cv2.imshow("img2zero",cikan2)

cv2.imshow("IMG 1",img1)
cv2.imshow("IMG 2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

