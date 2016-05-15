import numpy as np
import scipy.ndimage
import scipy.misc

class EdgeDetector(object):
    """Edge detector for images."""

    def __init__(self, img):
        """Construct an edge detector for an image."""
        self.suppressed = self.non_maximal_suppresion(*self.get_gradients(img))

    def get_gradients(self, img):
        """Computes gradients of a grayscale image.

        Finds gradient magnitudes and directions for self.img.

        Returns:
            G: A (height, width) float numpy array of gradient magnitudes.

            theta: A (height, width) float numpy array of gradient directions.
        """

        blurred = scipy.ndimage.filters.gaussian_filter(img, 2)
        Gx = scipy.ndimage.filters.sobel(blurred, axis=1)
        Gy = scipy.ndimage.filters.sobel(blurred, axis=0)
        G = np.sqrt(Gx**2 + Gy**2)
        theta = np.arctan2(Gy, Gx)

        return G, theta

    def non_maximal_suppresion(self, G, theta):
        """Performs non-maximal-suppression of gradients.

        Bins into 4 directions (up/down, left/right, both diagonals),
        and sets non-maximal elements in a 3x3 neighborhood to zero.

        Args:
            G: A (height, width) float numpy array of gradient magnitudes.

            theta: A (height, width) float numpy array of gradient directions.

        Returns:
            suppressed: A (height, width) float numpy array of suppressed
                gradient magnitudes.
        """

        right = 0
        down_right = 1
        down = 2
        down_left = 3

        def bin_direction(direction):
            direction = direction * 180 / np.pi
            if direction > 180.0:
                direction -= 180.0

            if direction < 22.5:
                return right
            elif direction < 67.5:
                return down_right
            elif direction < 112.5:
                return down
            elif direction < 157.5:
                return down_left
            else:
                return right

        height, width = G.shape
        mirror = np.zeros((height+2, width+2))
        mirror[1:height+1, 1:width+1] = G
        suppressed = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                direction = bin_direction(theta[y, x])
                y_m = y+1
                x_m = x+1

                if direction == right:
                    if (mirror[y_m, x_m] >= mirror[y_m, x_m+1]
                        and mirror[y_m, x_m] >= mirror[y_m, x_m-1]):
                        suppressed[y, x] = mirror[y_m, x_m]
                elif direction == down_right:
                    if (mirror[y_m, x_m] >= mirror[y_m+1, x_m+1]
                        and mirror[y_m, x_m] >= mirror[y_m-1, x_m-1]):
                        suppressed[y, x] = mirror[y_m, x_m]
                elif direction == down:
                    if (mirror[y_m, x_m] >= mirror[y_m+1, x_m]
                        and mirror[y_m, x_m] >= mirror[y_m-1, x_m]):
                        suppressed[y, x] = mirror[y_m, x_m]
                elif direction == down_left:
                    if (mirror[y_m, x_m] >= mirror[y_m+1, x_m-1]
                        and mirror[y_m, x_m] >= mirror[y_m-1, x_m+1]):
                        suppressed[y, x] = mirror[y_m, x_m]

        return suppressed

    def detect_edges(self, upper=None, lower=None):
        """Performs edge detection on a grayscale image.

        Performs Canny edge detection on img.

        Args:
            upper: Upper threshold (float)

            lower: Lower threshold (float)

        Returns:
            edges: A (heigh, width) float numpy array representing
                the edges of the image.
        """

        if upper == None:
            upper = 0.2
        if lower == None:
            lower = 0.4

        edges = self.suppressed.copy()
        median = np.median(edges[edges > 0.0])
        mean = np.mean(edges[edges > 0.0])
        edges[edges < upper*np.max(edges)] = 0.0
        edges[edges > 0.0] = 1.0
        return edges

