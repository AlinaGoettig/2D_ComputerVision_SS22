import numpy as np
from skimage import io
import util as ut
import math
import matplotlib.pyplot as plt
from skimage import util as skutil
import cv2
from skimage.transform import rotate
from skimage.feature import canny


def get_sign(img, version=1):
    if version == 1: # Gradient-Version
        # Kantendetektion
        img_1_gray = ut.rgb2gray(img)
        imgOut_gradient = ut.gradient(img_1_gray)
        
        # Rauschen ausbessern durch Morphologische Filter
        full = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

        img_circles = ut.dilate(imgOut_gradient, full, 1)
        image = (img_circles).astype('uint8')

        keypoints_circle = []
        keypoints_circle_1 = _detect_ellipse(image, 1000, 0.8, 1)
        keypoints_circle_2 = _detect_ellipse(image, 20000, 0.8, 1)
        for points in keypoints_circle_1:
            keypoints_circle.append(points)
        for points in keypoints_circle_2:
            keypoints_circle.append(points)
        
        circles = _just_circles(keypoints_circle, image)
        blobImg = _blobsOnImg(img, circles)
        zoomCircles = _zoom(blobImg, keypoints_circle)
        return zoomCircles

    elif version == 2: # Alexanders Canny-Version
        image_gray = ut.rgb2gray(img).astype('uint8')
        edges = cv2.Canny(image_gray, threshold1=100, threshold2=300)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(edges,kernel,iterations = 2)
        erosion = cv2.erode(dilate,kernel,iterations = 2)
        edged = np.invert(erosion)

        contours, _ =  cv2.findContours(edged, 1, 2)

        minThresE = 10000
        maxThresE = 150000
        elipse_list = []
        t_img = image_gray.copy()
        for cnt in contours:
            try:
                area = cv2.contourArea(cnt)
                if minThresE < area < maxThresE:
                    
                    ellipse = cv2.fitEllipse(cnt)
                    
                    x, y = ellipse[0]
                    a, b = ellipse[1] 
                    angle = ellipse[2]
                    elipse_list.append(((x, y), (a, b), angle))

                    a = a / 2
                    b = b / 2
            except:
                continue
        
        new_imges = []
        for el in elipse_list:
            new_img = _four_point_transform(img, el)
            new_imges.append(new_img)

            
        return new_imges

    elif version == 3: # Alinas Canny-Version
        img_1_gray = ut.rgb2gray(img)
        edges = canny(img_1_gray, sigma=3.0, low_threshold=0.25, high_threshold=0.8)

        full = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        edges = ut.dilate(edges, full, 1)

        edges = edges * 200
        edges = edges.astype('uint8')

        cannyE_1 = _detect_ellipse(edges, 1000, 0.8, 1)
        cannyE_2 = _detect_ellipse(edges, 20000, 0.8, 1)

        cannyPicPoints = []
        for points in cannyE_1:
            cannyPicPoints.append(points)
        for points in cannyE_2:
            cannyPicPoints.append(points)

        # Leave just the circles in the picture
        cannys = _just_circles(cannyPicPoints, edges)
        cannyImage = _blobsOnImg(img, cannys)
        zoomCannys = _zoom(cannyImage, cannyPicPoints)

        return zoomCannys
    else:
         return None

def match_signs(img, ref_img, version=1):
    if version == 1:
        #sift
        img1 = ut.rgb2gray(img).astype('uint8')
        img2 = ref_img.astype('uint8')


        sift = cv2.SIFT_create()
        sift = cv2.ORB_create()


        keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        return len(matches) , len(keypoints_2)
    
    elif version == 2:
        # template matching
        img1 = ut.rgb2gray(img).astype('uint8')
        ref_img = ref_img.astype('uint8')
        img_gray = cv2.resize(img1, ref_img.shape[::-1], interpolation = cv2.INTER_AREA)

        # create mask
        mask = np.zeros((ref_img.shape[0], ref_img.shape[1], 3), dtype=np.uint8)
        cv2.circle(mask, (int(ref_img.shape[0]/2), int(ref_img.shape[0]/2)), int(ref_img.shape[0]/2), (255, 255, 255), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # contrast stretching
        # new_img = cv2.bitwise_and(img_gray, img_gray, mask=mask)
        # new_img = ((new_img - new_img.min()) * (255 / new_img.max())).astype('uint8')
        # new_img = np.where((255 - new_img) < 100,255,new_img) # Expected results

        # ut.printImage(new_img)
        template_list = _create_template_list(ref_img)

        # matching
        res = []
        angle = 0
        for entry in template_list:
            res.append(_match_single_sign(img_gray, entry, mask))
        res = np.array(res)
        return res.max(), np.where(res == res.max())

    elif version == 3:
        # spectrum matching
        img1 = ut.rgb2gray(img).astype('uint8')
        ref_img = ref_img.astype('uint8')
        img_gray = cv2.resize(img1, ref_img.shape[::-1], interpolation=cv2.INTER_AREA)

        ref_img = 255 - ref_img
        mask = np.zeros((ref_img.shape[0], ref_img.shape[1], 3), dtype=np.uint8)
        cv2.circle(mask, (int(ref_img.shape[0] / 2), int(ref_img.shape[0] / 2)), int(ref_img.shape[0] / 2),
                   (255, 255, 255), thickness=-1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ref_img = 255 - cv2.bitwise_and(ref_img, ref_img, mask=mask)

        ref_spec = _get_horizontal_spectrum(ref_img)
        # ref_spec.sort()

        img_gray = 255 - img_gray
        img_gray = 255 - cv2.bitwise_and(img_gray, img_gray, mask=mask)

        test_spec = _get_horizontal_spectrum(img_gray)
        test_spec = _resize_histo(test_spec, len(ref_spec))
        # test_spec.sort()

        ref_cum = ut.cum_histo(ref_spec)
        test_cum = ut.cum_histo(test_spec)

        # cum_diff = ref_cum - test_cum
        # cum_diff = ref_spec / np.max(ref_spec) - test_spec / np.max(test_spec)
        # cum_diff = cv2.compareHist(ref_spec, test_spec, cv2.HISTCMP_CORREL)

        eps = 1e-10
        # compute the chi-squared distance
        # cum_diff = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(ref_spec, test_spec)])

        # minima = np.minimum(ref_spec, test_spec)
        # cum_diff = np.true_divide(np.sum(minima), np.sum(ref_spec))

        ref_count = 0
        ref_max = max(ref_spec)

        for i in ref_spec:
            if i > 0.9 * ref_max:
                ref_count += 1

        test_count = 0
        test_max = max(test_spec)

        for i in test_spec:
            if i > 0.8 * test_max:
                test_count += 1

        cum_diff = ref_count - test_count

        # match_result = max(cum_diff) + min(cum_diff) if min(cum_diff) > 0 else max(cum_diff) - min(cum_diff)
        match_result = np.mean(cum_diff)

        if True:
            # ut.printImage(ref_img)
            # ut.printImage(img)
            ut.printHisto(ref_spec)
            ut.printHisto(test_spec)
            # ut.printHisto(ref_cum)
            # ut.printHisto(test_cum)
            # ut.printHisto(cum_diff)
            # print(match_result)

        return abs(match_result)

        # if abs(match_result) < 0.4:
        #     return match_result
        # else:
        #     return False
    else:
        return None



##### helper functions #####

def _create_template_list(img):
    
    res_list = []
    for x in range(0, 360):
        img2 = rotate(img, x, False)
        img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        res_list.append(img2)
    return res_list

def _match_single_sign(img, ref_img, mask):
    # matching
    img = img.astype('uint8')
    res = cv2.matchTemplate(img,ref_img,cv2.TM_CCOEFF_NORMED, mask=mask)
    # res = cv2.matchTemplate(img.astype('uint8'),ref_img.astype('uint8'),cv2.TM_CCORR_NORMED, mask=mask)
    # res = cv2.matchTemplate(img.astype('uint8'),ref_img.astype('uint8'),cv2.TM_SQDIFF_NORMED, mask=mask)
    return res.max()

def _detect_ellipse(img, minArea, minCircularity, maxCircularity):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = False
    params.filterByArea = True
    params.minArea = minArea #20000 #1000
    params.maxArea = 100000000
    params.filterByCircularity = True
    params.minCircularity = minCircularity
    params.maxCircularity = maxCircularity
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    #params.minInertiaRatio = 0.8
    params.minInertiaRatio = 0.3
    params.minThreshold = 1
    params.maxThreshold = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)

    text = "Number of Circular Blobs: " + str(len(keypoints))
    print(text)

    return keypoints

def _blobsOnImg(img, blobPic):
    # Blob Maske auf Ursprüngliches Bild anwenden
    blank = np.zeros(img.shape, int)
    blank = 255-blank
    #inv = 255-blobPic
    width = img.shape[0]
    height = img.shape[1]
    for i in range(width):
        for j in range(height):
            if (blobPic[i][j] != 255):
                blank[i][j] = img[i][j]

    return blank

def _just_circles(keyP, img):
    img = img.copy()
    empty_img = np.zeros([img.shape[0], img.shape[1]],dtype=np.uint8)
    empty_img.fill(255)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(empty_img, keyP, blank, (0,0,0),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Fill circles
    gray = cv2.cvtColor(blobs, cv2.COLOR_BGR2GRAY)
    #ut.printImage(gray)
    th, im_th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # NOTE: the size needs to be 2 pixels bigger on each side than the input image
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground
    circles_inv = im_th | im_floodfill_inv
    circles_pic = 255-circles_inv

    return circles_pic

def _zoom(img, keypoints):
    set = list()
    for kp in keypoints:
        minX = int(kp.pt[0]-kp.size/2)
        maxX = int(kp.pt[0]+kp.size/2+1)
        minY = int(kp.pt[1]-kp.size/2)
        maxY = int(kp.pt[1]+kp.size/2+1)
        # print(minX, maxX, minY, maxY)

        set.append(img[minY:maxY,minX:maxX])
    return set

def _get_vertical_spectrum(ref):
    spec = np.zeros(len(ref))
    for i in range(len(ref)):
        for pxl in ref[i]:
            if pxl < 150:
                spec[i] += 1
    return spec

def _get_horizontal_spectrum(ref):
    spec = np.zeros(len(ref))
    for j in range(len(ref)):
        for i in range(len(ref[0])):
            if ref[j][i] < 150:
                spec[i] += 1
    return spec

def _resize_histo(histo, size):
    result = np.zeros(size)
    for i in range(len(histo)):
        idx = round(i / len(histo) * size)
        idx -= 1 if idx >= size else 0
        result[idx] += histo[i]
    for i in range(1, len(result) - 1):
        if result[i] > 1.8 * result[-i] or result[i] > 1.8 * result[i + 1] or result[i] > 1.8 * result[i - 1]:
            result[i] = result[i + 1]
    return result

def _rotate(origin, point, angle):
	#Rotate a point counterclockwise by a given angle around a given origin.
	#The angle should be given in radians.
	x = origin[0] + math.cos(angle) * (point[0] - origin[0]) - math.sin(angle) * (point[1] - origin[1])
	y = origin[1] + math.sin(angle) * (point[0] - origin[0]) + math.cos(angle) * (point[1] - origin[1])
	return [x, y]

def _four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	x = pts[0][0]
	y = pts[0][1]
	a = pts[1][0] / 2
	b = pts[1][1] / 2
	angle = pts[2]
	points = (_rotate((0,0), (a, 0), math.radians(angle) ),
			  _rotate((0,0), (-a, 0), math.radians(angle)),
			  _rotate((0,0), (0,b), math.radians(angle)),
			  _rotate((0,0), (0,-b), math.radians(angle)))

	elipse_pts = np.array([
		[points[0][0]+ x, points[0][1] + y],
		[points[1][0]+ x, points[1][1] + y],
		[points[2][0]+ x, points[2][1] + y],
		[points[3][0]+ x, points[3][1] + y]], dtype = "float32")

	maxWidth = 180
	maxHeight = 180

	dst_pts = (
		_rotate((0, 0), (90, 0), math.radians(angle)),
		_rotate((0, 0), (-90, 0), math.radians(angle)),
		_rotate((0, 0), (0, 90), math.radians(angle)),
		_rotate((0, 0), (0, -90), math.radians(angle)))

	dst = np.array([
		[dst_pts[0][0]+ 90, dst_pts[0][1] + 90],
		[dst_pts[1][0]+ 90, dst_pts[1][1] + 90],
		[dst_pts[2][0]+ 90, dst_pts[2][1] + 90],
		[dst_pts[3][0]+ 90, dst_pts[3][1] + 90]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(elipse_pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped