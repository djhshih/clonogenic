import numpy as np
import cv2 as cv

img = cv.imread("EPSON005.TIF", cv.IMREAD_GRAYSCALE)
img = cv.medianBlur(img, 5)

# convert to grayscale
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

print(img.shape)

height= img.shape[0]
width = img.shape[1]

plate_nrows = 2
plate_ncols = 3

max_radius = int(min((width / plate_ncols), (height / plate_nrows)) / 2)
print(max_radius)

wells = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 2, max_radius,
    param1 = 100, param2 = 100,
    minRadius = int(max_radius / 2), maxRadius = max_radius)[0]
wells = np.uint16(np.around(wells))
print(wells)

# TODO sort wells by coordinates

for w in wells:
    x0, y0, r = w
    # draw the perimeter
    cv.circle(cimg, (x0, y0), r, (0, 255, 0), 2)
    # draw the center
    cv.circle(cimg, (x0, y0), 2, (0, 0, 255), 3)

cv.imshow("Detected wells", cimg)
cv.waitKey(0)


def process_well_hough(img, well):
    x0, y0, r = well
    wimg = img[(y0-r):(y0+r), (x0-r):(x0+r)]
    wcimg = cimg[(y0-r):(y0+r), (x0-r):(x0+r)]

    circles = cv.HoughCircles(wimg, cv.HOUGH_GRADIENT, 1, 3,
        param1 = 30, param2 = 20,
        minRadius = 3, maxRadius = 20)
    print(circles)

    if circles is not None:
        circles = circles[0]
        circles = np.uint16(np.around(circles))
        print(circles)
        for c in circles:
            x0, y0, r = c
            cv.circle(wcimg, (x0, y0), r, (255, 0, 0), 2)

    cv.imshow("well", wcimg)
    cv.waitKey(0)

#process_well_hough(wells[0])


well = wells[0]
x0, y0, r = well

wimg = img[(y0-r):(y0+r), (x0-r):(x0+r)]
wimg = cv.bitwise_not(wimg)

def apply_circular_mask(wimg, border=10):
    mask = np.zeros(wimg.shape, np.uint8)
    cv.circle(mask, (r, r), r-border, 255, -1)
    return cv.bitwise_and(wimg, mask)

cv.imshow("well", wimg)
cv.waitKey(0)

#wimg = apply_circular_mask(wimg)

bimg = cv.GaussianBlur(wimg, (7, 7), 0)
cv.imshow("blur", bimg)
cv.waitKey(0)

lg = cv.Laplacian(bimg, cv.CV_16S, 11)
#kernel = np.ones((3,3))
kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
# use erode to find min of neighbourhood
minlg = cv.morphologyEx(lg, cv.MORPH_ERODE, kernel)
# use dilate to find max of neighbourhood
maxlg = cv.morphologyEx(lg, cv.MORPH_DILATE, kernel)
# zero crossing occurs at a positive pixel with an adjacent negative pixel or
# a negative pixel with an adjacent positive pixel 
zcross = np.logical_or(np.logical_and(minlg < 0, lg > 0), np.logical_and(maxlg > 0, lg < 0))

cv.imshow("zero_crossing", zcross.astype(np.uint8)*255)
cv.waitKey(0)

fimg = wimg.astype(np.float)

m = cv.blur(fimg, (3, 3))
m2 = cv.blur(np.multiply(fimg, fimg), (3, 3))
sd = cv.sqrt(m2 - m*m)

cv.imshow("sd", sd)
cv.waitKey(0)

# edge occurs at Laplacian zero-crossings in regions of high local variance
ledge = np.logical_and(sd > 2, zcross).astype(np.uint8)*255
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)

ledge = cv.morphologyEx(ledge, cv.MORPH_CLOSE, np.ones((3,3)))
#ledge = cv.medianBlur(ledge, 5)
cv.imshow("laplacian_edge", ledge)
cv.waitKey(0)

cedge = cv.Canny(bimg, 30, 15, 3)
cv.imshow("canny_edge", cedge)
cv.waitKey(0)



#lg_abs = cv.convertScaleAbs(lg, alpha = 255/(2*np.max(lg)))
#lg_abs = apply_circular_mask(lg_abs)
#lg = cv.threshold(lg_abs, 10, 255, cv.THRESH_BINARY)[1]

#print(np.mean(lg_abs))
print(zcross)

#cv.imshow("lglacian", lg == 0)
#cv.imshow("lglacian", lg_abs)

#dimg = wimg - bimg
#cv.imshow("diff1", dimg)
#cv.waitKey(0)

#dimg = bimg - wimg
#cv.imshow("diff", dimg)
#cv.waitKey(0)

timg = cv.adaptiveThreshold(wimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -2)
cv.imshow("athreshold", timg)
cv.waitKey(0)

timg = cv.medianBlur(timg, 5)
#timg = cv.morphologyEx(timg, cv.MORPH_OPEN, np.ones((3,3)))
cv.imshow("athreshold", timg)
cv.waitKey(0)


cv.destroyAllWindows()
