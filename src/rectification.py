import cv2
from pad import plot_image

def find_contours(
    image,
    verbose=False
):
    blurred_image = cv2.blur(image, (2, 2))
    edges = cv2.Canny(blurred_image, 40, 150)
    edges_2, contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    black_and_white_edges = cv2.cvtColor(
        edges_2,
        cv2.COLOR_GRAY2RGB
    )
    if verbose:
        plot_image(blurred_image, title="Blurred")
        plot_image(edges, title="Edges")
        plot_image(edges_2, title="OpenCV is dumb. Why does it return the same thing?")
        plot_image(black_and_white_edges, title="Black and White Edges")
    return black_and_white_edges, contours, hierarchy

def select_markers(contours, hierarchy, verbose=False, very_verbose=False):
    """
    Based off of Galen's code. This doesn't work. It returns nothing.
    """
    markers = []
    for i, contour in enumerate(contours):
        if very_verbose:
            print(f"i: {i}")
            print(f"hierarchy[0][i][2]:\n{hierarchy[0][i][2]}")
            print(f"contour:\n{contour}")
        depth = 0
        while hierarchy[0][i][2] != -1:
            i = hierarchy[0][i][2]
            # print(f"other contour:\n{contour}")
            depth += 1
        if very_verbose:
            print(f"depth: {depth}")
        if hierarchy[0][i][2] != -1:
            depth += 1
        if depth >= 3: # Sometimes 5?
            markers.append(i)
    if verbose:
        print(f"Markers: {markers}")
    return markers

def find_fiducials(contours, hierarchy):
    fiducials = []
    markers = select_markers(contours, hierarchy)