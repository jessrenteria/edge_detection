import numpy as np
import scipy.ndimage
import scipy.misc

class EdgeDetector(object):
    """Edge detector for images."""

    def __init__(self, img):
        """Construct an edge detector for an image."""
        self.suppressed = self.non_maximal_suppression(*self.get_gradients(img))

    def get_gradients(self, img):
        """Computes gradients of a grayscale image.

        Finds gradient magnitudes and directions for self.img.

        Returns:
            G: A (height, width) float numpy array of gradient magnitudes.

            theta: A (height, width) float numpy array of gradient directions.
        """

        blurred = scipy.ndimage.gaussian_filter(img, 2)
        Gx = scipy.ndimage.sobel(blurred, axis=1)
        Gy = scipy.ndimage.sobel(blurred, axis=0)
        G = np.sqrt(Gx**2 + Gy**2)
        theta = np.arctan2(Gy, Gx)

        return G, theta

    def non_maximal_suppression(self, G, theta):
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

        theta *= 180.0 / np.pi
        theta[theta > 180.0] -= 180.0
        hits = np.zeros_like(G, dtype=bool)
        correlate = scipy.ndimage.correlate
        correlate1d = scipy.ndimage.correlate1d
        convolve = scipy.ndimage.convolve
        convolve1d = scipy.ndimage.convolve1d

        kernel = np.array([0.0, 1.0, -1.0])
        mask = np.logical_or(theta < 22.5, theta > 157.5)
        hits[mask] = np.logical_and(correlate1d(G, kernel, axis=-1)[mask] >= 0.0,
                                    convolve1d(G, kernel, axis=-1)[mask] >= 0.0)

        mask = np.logical_and(theta >= 67.5, theta < 112.5)
        hits[mask] = np.logical_and(correlate1d(G, kernel, axis=0)[mask] >= 0.0,
                                    convolve1d(G, kernel, axis=0)[mask] >= 0.0)

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, -1.0]])
        mask = np.logical_and(theta >= 22.5, theta < 67.5)
        hits[mask] = np.logical_and(correlate(G, kernel)[mask] >= 0.0,
                                    convolve(G, kernel)[mask] >= 0.0)

        kernel = np.array([[0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [-1.0, 0.0, 0.0]])
        mask = np.logical_and(theta >= 112.5, theta < 157.5)
        hits[mask] = np.logical_and(correlate(G, kernel)[mask] >= 0.0,
                                    convolve(G, kernel)[mask] >= 0.0)

        suppressed = G.copy()
        suppressed[np.logical_not(hits)] = 0.0

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

