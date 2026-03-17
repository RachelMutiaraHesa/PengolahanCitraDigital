import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim

# =========================
# 1. LOAD GAMBAR
# =========================
img = cv2.imread('ZOOTOPIA.jpEg')  
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))

# =========================
# 2. TAMBAH NOISE
# =========================
def add_gaussian_noise(image):
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(image, prob=0.02):
    noisy = image.copy()
    rnd = np.random.rand(*image.shape[:2])

    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy

def add_speckle(image):
    gauss = np.random.randn(*image.shape)
    noisy = image + image * gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

gaussian_noise = add_gaussian_noise(img)
sp_noise = add_salt_pepper(img)
speckle_noise = add_speckle(img)

# =========================
# 3. FILTERING
# =========================
def apply_filters(image):
    results = {}

    # Mean Filter
    results['Mean 3x3'] = cv2.blur(image, (3,3))
    results['Mean 5x5'] = cv2.blur(image, (5,5))

    # Gaussian Filter
    results['Gaussian σ=1'] = cv2.GaussianBlur(image, (5,5), 1)
    results['Gaussian σ=2'] = cv2.GaussianBlur(image, (5,5), 2)

    # Median Filter
    results['Median 3x3'] = cv2.medianBlur(image, 3)
    results['Median 5x5'] = cv2.medianBlur(image, 5)

    # Min Filter (Erosi)
    kernel = np.ones((3,3), np.uint8)
    results['Min Filter'] = cv2.erode(image, kernel)

    return results

# =========================
# 4. METRIK
# =========================
def mse(original, processed):
    return np.mean((original - processed) ** 2)

def psnr(original, processed):
    mse_val = mse(original, processed)
    if mse_val == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse_val))

def compute_ssim(original, processed):
    return ssim(original, processed, channel_axis=2)

# =========================
# 5. EVALUASI
# =========================
def evaluate(noisy_img, name):
    print(f"\n===== {name} =====")
    filtered = apply_filters(noisy_img)

    for key, img_f in filtered.items():
        start = time.time()

        mse_val = mse(img, img_f)
        psnr_val = psnr(img, img_f)
        ssim_val = compute_ssim(img, img_f)

        end = time.time()

        print(f"{key:15} | MSE={mse_val:.2f} | PSNR={psnr_val:.2f} | SSIM={ssim_val:.4f} | Time={end-start:.5f}s")

# Jalankan evaluasi
evaluate(gaussian_noise, "Gaussian Noise")
evaluate(sp_noise, "Salt & Pepper Noise")
evaluate(speckle_noise, "Speckle Noise")

# =========================
# 6. VISUALISASI
# =========================
def show_images(title, images):
    n = len(images)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(15, 8))
    for i, (name, img_show) in enumerate(images.items()):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_show)
        plt.title(name)
        plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# =========================
# 7. TAMPILKAN HASIL
# =========================

# Original + Noise
show_images("Original & Noise", {
    "Original": img,
    "Gaussian": gaussian_noise,
    "Salt & Pepper": sp_noise,
    "Speckle": speckle_noise
})

# Gaussian + Filter
gaussian_results = apply_filters(gaussian_noise)
gaussian_results["Noisy"] = gaussian_noise
show_images("Gaussian Noise + Filtering", gaussian_results)

# Salt & Pepper + Filter
sp_results = apply_filters(sp_noise)
sp_results["Noisy"] = sp_noise
show_images("Salt & Pepper Noise + Filtering", sp_results)

# Speckle + Filter
speckle_results = apply_filters(speckle_noise)
speckle_results["Noisy"] = speckle_noise
show_images("Speckle Noise + Filtering", speckle_results)