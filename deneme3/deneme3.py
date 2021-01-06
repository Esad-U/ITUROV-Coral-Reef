import matplotlib.pyplot as plt
import cv2
import numpy as np

img1 = cv2.imread("Reef1.png")
img2 = cv2.imread("Reef2.png")

img1rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000)

keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

search_params=dict(checks=32)

matcher = cv2.FlannBasedMatcher(index_params, search_params)

matches = matcher.knnMatch(descriptors1, descriptors2, 2)



ratio = 0.7
goodMatches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]
print('good matches:%d/%d' %(len(goodMatches),len(matches)))

fig = plt.figure(figsize=(50, 50))
imMatches = cv2.drawMatches(img1rgb, keypoints1, img2rgb, keypoints2, goodMatches, None,
                     flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

points1 = np.zeros((len(goodMatches), 2), dtype=np.float32)
points2 = np.zeros((len(goodMatches), 2), dtype=np.float32)

for i, match in enumerate(goodMatches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)


height, width, channels = img2.shape
input_transformed = cv2.warpPerspective(img1, h, (width, height))
input_transformed_rgb = cv2.cvtColor(input_transformed, cv2.COLOR_BGR2RGB)


# Show images
ax = fig.add_subplot(2, 2, 1)
ax.set_title('original first image')
plt.imshow(img1rgb)

ax = fig.add_subplot(2, 2, 2)
ax.set_title('Matched features')
plt.imshow(imMatches)

ax = fig.add_subplot(2, 2, 3)
ax.set_title('warped')
plt.imshow(input_transformed_rgb)
plt.show()


cv2.imshow("Reef1",img1)
cv2.imshow("Reef2",img2)

def kontur_alma(img):

    lower_beyaz = np.array([35, 30, 175])
    upper_beyaz = np.array([75, 100, 255])

    blur = cv2.blur(img,(3,3))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_beyaz, upper_beyaz)

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i],False))

    bg = np.zeros((img.shape[0],img.shape[1],3),np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(bg,contours,i,(255,255,255),1,8)
        cv2.drawContours(img,hull,i,(0,255,0),1,8)




    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])


    mask2 = cv2.inRange(hsv, lower_pembe, upper_pembe)

    contours2,hierarchy = cv2.findContours(mask2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    hull = []

    for i in range(len(contours2)):
        hull.append(cv2.convexHull(contours2[i],False))


    for i in range(len(contours2)):
        cv2.drawContours(bg,contours2,i,(0,0,255),1,8)
        cv2.drawContours(img,hull,i,(0,0,255),1,8)
    return bg

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
    lower_beyaz = np.array([83, 16, 140])
    upper_beyaz = np.array([130, 90, 255])

    hsv2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv2, lower_beyaz, upper_beyaz)
    return mask2

def mask_alma_pembe(img):
    lower_pembe = np.array([105, 55, 66])
    upper_pembe = np.array([179, 255, 255])

    hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv1, lower_pembe, upper_pembe)
    return mask1

def ters_alma(img):
    Height = img.shape[0]
    Width = img.shape[1]
    Channels = img.shape[2]
    Size = (Height,Width,Channels)
    new_img = np.zeros(Size, np.uint8)

    for x in range(0,Height):
        for y in range(0,Width):
            for c in range(0, Channels):
                new_img[x,y,c] = 255 - img[x,y,c]
    return new_img

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
        if area < 100:
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


cv2.imshow("Maske 1 Beyaz",mask_alma_beyaz(input_transformed))
cv2.imshow("Maske 1 Pembe",mask_alma_pembe(input_transformed))

cv2.imshow("Maske 2 Beyaz",mask_alma_beyaz(img2))
cv2.imshow("Maske 2 Pembe",mask_alma_pembe(img2))

mask_beyaz_img2 = mask_alma_beyaz(img2)
#cv2.imshow("Beyaz Mask 2", mask_beyaz_img2)
mask_beyaz_it = mask_alma_beyaz(input_transformed)
#cv2.imshow("Beyaz Mask 1", mask_beyaz_it)
sonuc = bit_xor(mask_beyaz_img2,mask_beyaz_it,img2)
#cv2.imshow("XOR Beyaz", sonuc[0])
cv2.imshow("XOR BLUR Beyaz", sonuc[1])
cv2.imshow("SONUC Beyaz", sonuc[2])



mask_pembe_img2 = mask_alma_pembe(img2)
#cv2.imshow("Pembe Mask 2",mask_pembe_img2)
mask_pembe_it = mask_alma_pembe(input_transformed)
#cv2.imshow("Pembe Mask 1",mask_pembe_it)
sonuc2 = bit_xor(mask_pembe_img2, mask_pembe_it,img2)
#cv2.imshow("XOR Pembe", sonuc2[0])
cv2.imshow("XOR BLUR Pembe", sonuc2[1])
cv2.imshow("SONUC Pembe", sonuc2[2])


cv2.imshow("imgSon",img2)

cv2.imshow("Wraped",input_transformed)

#cv2.imshow("Bitwise Reef2",bitwise_alma(img2))

#cv2.imshow("Ters Wraped",ters_alma(bitwise_alma(input_transformed)))

#top = cv2.add(img2rgb,ters_alma(bitwise_alma(input_transformed)))
#cv2.imshow("Toplam",top)

added = cv2.addWeighted(bitwise_alma(input_transformed),0.5,img2,0.5,0)
added = cv2.resize(added,(506,480))
cv2.imshow("Added", added)

cv2.waitKey(0)
cv2.destroyAllWindows()