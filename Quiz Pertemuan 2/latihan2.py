# ===============================
# IMPORT LIBRARY
# ===============================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# FUNGSI SIMULASI ALIASING
# ===============================
def simulate_image_aliasing(image, downsampling_factors):
    """
    Simulate aliasing by downsampling image

    Parameters:
    image : Input image (BGR or Grayscale)
    downsampling_factors : list of integers (2, 4, 8, ...)

    Returns:
    Dictionary containing downsampled images and aliasing analysis
    """

    if image is None:
        raise ValueError("Input image is empty or not loaded.")

    results = {}

    # Ubah ke grayscale agar aliasing terlihat jelas
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    for factor in downsampling_factors:
        # Downsampling tanpa anti-aliasing
        downsampled = gray[::factor, ::factor]

        # Upsampling kembali ke ukuran asli
        restored = cv2.resize(
            downsampled,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # Hitung error aliasing
        diff = cv2.absdiff(gray, restored)

        results[factor] = {
            'downsampled': downsampled,
            'restored': restored,
            'mean_error': np.mean(diff),
            'std_error': np.std(diff),
            'analysis': (
                f"Downsampling x{factor} menyebabkan hilangnya detail frekuensi tinggi "
                f"dan muncul distorsi aliasing seperti tepi bergerigi."
            )
        }

    return results

# ===============================
# LOAD GAMBAR
# ===============================
image = cv2.imread("kucing.jpg")

if image is None:
    print("Gambar kucing.jpg tidak ditemukan. Upload dulu filenya.")
else:
    print("Gambar berhasil dimuat.")

    # ===============================
    # JALANKAN SIMULASI
    # ===============================
    factors = [2, 4, 8]
    aliasing_results = simulate_image_aliasing(image, factors)

    # ===============================
    # VISUALISASI OUTPUT
    # ===============================
    plt.figure(figsize=(12, 8))

    for i, factor in enumerate(factors):
        # Downsampled
        plt.subplot(3, len(factors), i+1)
        plt.imshow(aliasing_results[factor]['downsampled'], cmap='gray')
        plt.title(f"Downsample x{factor}")
        plt.axis("off")

        # Restored
        plt.subplot(3, len(factors), i+1+len(factors))
        plt.imshow(aliasing_results[factor]['restored'], cmap='gray')
        plt.title(f"Restored x{factor}")
        plt.axis("off")

        # Error
        plt.subplot(3, len(factors), i+1+2*len(factors))
        error_img = cv2.absdiff(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            aliasing_results[factor]['restored']
        )
        plt.imshow(error_img, cmap='hot')
        plt.title("Aliasing Error")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ===============================
    # OUTPUT ANALISIS NUMERIK
    # ===============================
    for factor in factors:
        print(f"\n=== Downsampling x{factor} ===")
        print("Mean Error :", aliasing_results[factor]['mean_error'])
        print("Std Error  :", aliasing_results[factor]['std_error'])
        print("Analysis   :", aliasing_results[factor]['analysis'])