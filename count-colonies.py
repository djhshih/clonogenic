import numpy as np
import cv2 as cv

img = cv.imread("EPSON005.TIF", 0)
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


well = wells[0]
print(well)

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

cv.imshow("Well 1", wcimg)
cv.waitKey(0)

cv.destroyAllWindows()
