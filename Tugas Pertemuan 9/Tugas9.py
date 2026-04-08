import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD GAMBAR
# =========================
img1 = cv2.imread('KUPU-KUPU.jpeg', 0)   # bimodal
img2 = cv2.imread('LAMPU.jpeg', 0)  # iluminasi tidak merata
img3 = cv2.imread('KOIN.jpeg', 0)   # overlapping

images = {
    "Bimodal": img1,
    "Uneven": img2,
    "Overlapping": img3
}

# =========================
# THRESHOLDING
# =========================
def thresholding(image):
    _, global_th = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(image, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return global_th, otsu, adaptive

# =========================
# EDGE DETECTION
# =========================
def edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    prewitt_kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    prewitt_kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewittx = cv2.filter2D(image, -1, prewitt_kernelx)
    prewitty = cv2.filter2D(image, -1, prewitt_kernely)
    prewitt = cv2.add(prewittx, prewitty)

    canny = cv2.Canny(image, 50, 150)

    return sobel, prewitt, canny

# =========================
# VISUALISASI
# =========================
for name, img in images.items():

    g, o, a = thresholding(img)
    s, p, c = edge_detection(img)

    fig, ax = plt.subplots(2, 4, figsize=(15,8))

    ax[0,0].imshow(img, cmap='gray')
    ax[0,0].set_title(f"{name} - Original")

    ax[0,1].imshow(g, cmap='gray')
    ax[0,1].set_title("Global")

    ax[0,2].imshow(o, cmap='gray')
    ax[0,2].set_title("Otsu")

    ax[0,3].imshow(a, cmap='gray')
    ax[0,3].set_title("Adaptive")

    ax[1,0].imshow(s, cmap='gray')
    ax[1,0].set_title("Sobel")

    ax[1,1].imshow(p, cmap='gray')
    ax[1,1].set_title("Prewitt")

    ax[1,2].imshow(c, cmap='gray')
    ax[1,2].set_title("Canny")

    ax[1,3].axis('off')

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.show()

# =========================
# CONNECTED COMPONENT (KHUSUS KOIN)
# =========================
_, binary = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels = cv2.connectedComponents(binary)

print("Jumlah objek terdeteksi (koin): ", num_labels - 1)