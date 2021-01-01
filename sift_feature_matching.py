import cv2

img1 = cv2.imread("../Reef1.png")
img2 = cv2.imread("../Reef2.png")
#img1 = cv2.resize(img1,(640,380))
#img2 = cv2.resize(img2,(640,380))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Features
sift = cv2.xfeatures2d.SIFT_create()
kp_img1, desc_img1 = sift.detectAndCompute(gray1, None)
img1 = cv2.drawKeypoints(img1,kp_img1,gray1)
kp_img2, desc_img2 = sift.detectAndCompute(gray2, None)
img2 = cv2.drawKeypoints(img2, kp_img2, gray2)
#Feature Matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(desc_img1, desc_img2, k=2)

good = []
for m,n in matches:
    if m.distance < .5 * n.distance:    #n.distance'ın başındaki değer 0-1 arasında değişebilir
        good.append(m)

img3 = cv2.drawMatches(img1,kp_img1,img2,kp_img2,good,img2)

#cv2.imshow("img1",img1)
#cv2.imshow("img2",img2)
cv2.imshow("img3",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
