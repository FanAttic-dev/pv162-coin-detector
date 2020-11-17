
def detect_circles_skiimage(image_rgb):
    image_gray = color.rgb2gray(image_rgb)

    # Detect edges
    edges = canny(image_gray, sigma=3)

    # Detect two radii
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, threshold=0.35)

    # Draw them
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=image_rgb.shape)
        image_rgb[circy, circx] = (220, 20, 20)

    #ax.imshow(image_rgb, cmap=plt.cm.gray)
    #plt.show()
    return image_rgb

def detect_ellipses():
    # Load picture, convert to grayscale and detect edges
    #image_rgb = data.coffee()[0:220, 160:420]
    #image_gray = color.rgb2gray(image_rgb)
    image_gray = data.coins()[160:230, 70:150]
    image_rgb = color.gray2rgb(image_gray)
    edges = canny(image_gray, sigma=4.0)
    #plt.imshow(edges)
    #plt.show()

    print(np.shape(image_gray))
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=10)
    print(result)
    result.sort(order='accumulator')


    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()
