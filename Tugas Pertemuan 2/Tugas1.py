#24343110/Rachel Mutiara Hesa
# ==========================================
# PROYEK MINI: KONVERSI MODEL WARNA
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# LOAD GAMBAR 
# ==========================================
image_paths = [
    "cahaya.jpeg",
    "normal.jpeg",
    "redup.jpeg"
]

# ==========================================
# FUNGSI KUANTISASI UNIFORM
# ==========================================
def uniform_quantization(img, levels=16):
    step = 256 // levels
    quantized = (img // step) * step
    return quantized

# ==========================================
# FUNGSI KUANTISASI NON-UNIFORM (KMEANS)
# ==========================================
def kmeans_quantization(img, k=16):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape((img.shape))
    return result

# ==========================================
# PROSES SETIAP GAMBAR
# ==========================================
for path in image_paths:

    print("\nMemproses:", path)
    image = cv2.imread(path)

    # Konversi BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ==============================
    # KONVERSI MODEL WARNA
    # ==============================
    start = time.time()

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    end = time.time()
    print("Waktu konversi warna:", end - start, "detik")

    # ==============================
    # KUANTISASI
    # ==============================
    uniform_rgb = uniform_quantization(image_rgb, 16)
    kmeans_rgb = kmeans_quantization(image_rgb, 16)

    # ==============================
    # HITUNG MEMORI
    # ==============================
    original_size = image_rgb.nbytes
    uniform_size = uniform_rgb.nbytes
    kmeans_size = kmeans_rgb.nbytes

    print("Ukuran asli:", original_size, "bytes")
    print("Ukuran uniform:", uniform_size, "bytes")
    print("Ukuran kmeans:", kmeans_size, "bytes")

    print("Rasio kompresi (uniform):", original_size / uniform_size)
    print("Rasio kompresi (kmeans):", original_size / kmeans_size)

    # ==============================
    # VISUALISASI
    # ==============================
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.imshow(image_rgb)
    plt.title("Original")

    plt.subplot(2,3,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")

    plt.subplot(2,3,3)
    plt.imshow(hsv)
    plt.title("HSV")

    plt.subplot(2,3,4)
    plt.imshow(lab)
    plt.title("LAB")

    plt.subplot(2,3,5)
    plt.imshow(uniform_rgb)
    plt.title("Uniform Quantization")

    plt.subplot(2,3,6)
    plt.imshow(kmeans_rgb)
    plt.title("KMeans Quantization")

    plt.tight_layout()
    plt.show()

    # ==============================
    # HISTOGRAM
    # ==============================
    plt.figure()
    plt.title("Histogram RGB Original")
    plt.hist(image_rgb.ravel(), bins=256)
    plt.show()

    plt.figure()
    plt.title("Histogram RGB Uniform")
    plt.hist(uniform_rgb.ravel(), bins=256)
    plt.show()

    plt.figure()
    plt.title("Histogram RGB KMeans")
    plt.hist(kmeans_rgb.ravel(), bins=256)
    plt.show()

print("\nSelesai semua proses.")