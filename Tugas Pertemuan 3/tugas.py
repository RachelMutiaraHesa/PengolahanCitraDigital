import cv2
import numpy as np
import time
import math

# =========================================
# FUNGSI EVALUASI
# =========================================

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

# =========================================
# LOAD GAMBAR
# =========================================

ref = cv2.imread("Lurus.jpeg")
target = cv2.imread("Miring.jpeg")

if ref is None or target is None:
    print("Gambar tidak ditemukan! Pastikan nama file benar.")
    exit()

# Samakan ukuran
target = cv2.resize(target, (ref.shape[1], ref.shape[0]))

ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

h, w = ref_gray.shape

# =========================================
# 1. TRANSLASI
# =========================================

T = np.float32([[1, 0, 50],
                [0, 1, 30]])

translated = cv2.warpAffine(target_gray, T, (w, h))

# =========================================
# 2. ROTASI
# =========================================

R = cv2.getRotationMatrix2D((w//2, h//2), -90, 1.0)
rotated = cv2.warpAffine(target_gray, R, (w, h))

# =========================================
# 3. SCALING
# =========================================

S = np.float32([[0.8, 0, 0],
                [0, 0.8, 0]])

scaled = cv2.warpAffine(target_gray, S, (w, h))

# =========================================
# 4. AFFINE (3 TITIK)
# =========================================

pts1 = np.float32([[50,50], [200,50], [50,200]])
pts2 = np.float32([[10,100], [200,50], [100,250]])

M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(target_gray, M_affine, (w, h))

# =========================================
# 5. PERSPEKTIF (4 TITIK)
# =========================================

pts1_p = np.float32([[0,0], [w,0], [0,h], [w,h]])
pts2_p = np.float32([[50,50], [w-50,30], [30,h-30], [w-30,h-50]])

M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
perspective = cv2.warpPerspective(target_gray, M_persp, (w, h))

# =========================================
# 6. INTERPOLASI (ROTASI)
# =========================================

print("\n=== HASIL EVALUASI INTERPOLASI ===")

interpolations = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC
}

for name, method in interpolations.items():
    start = time.time()
    result = cv2.warpAffine(target_gray, R, (w, h), flags=method)
    end = time.time()

    mse_val = mse(ref_gray, result)
    psnr_val = psnr(ref_gray, result)
    comp_time = end - start

    print(f"\n{name}")
    print(f"MSE   : {mse_val:.4f}")
    print(f"PSNR  : {psnr_val:.4f}")
    print(f"Waktu : {comp_time:.6f} detik")

# =========================================
# SIMPAN HASIL INTERPOLASI TERPISAH
# =========================================

nearest_img = cv2.warpAffine(target_gray, R, (w, h), flags=cv2.INTER_NEAREST)
bilinear_img = cv2.warpAffine(target_gray, R, (w, h), flags=cv2.INTER_LINEAR)
bicubic_img = cv2.warpAffine(target_gray, R, (w, h), flags=cv2.INTER_CUBIC)

cv2.imwrite("nearest.jpg", nearest_img)
cv2.imwrite("bilinear.jpg", bilinear_img)
cv2.imwrite("bicubic.jpg", bicubic_img)

# =========================================
# SIMPAN SEMUA HASIL
# =========================================

cv2.imwrite("translated.jpg", translated)
cv2.imwrite("rotated.jpg", rotated)
cv2.imwrite("scaled.jpg", scaled)
cv2.imwrite("affine.jpg", affine)
cv2.imwrite("perspective.jpg", perspective)

print("\nSemua file hasil berhasil disimpan.")

# =========================================
# TAMPILKAN DI LAYAR
# =========================================

cv2.imshow("Referensi", ref_gray)
cv2.imshow("Target", target_gray)
cv2.imshow("Translated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Scaled", scaled)
cv2.imshow("Affine", affine)
cv2.imshow("Perspective", perspective)

cv2.imshow("Nearest Neighbor", nearest_img)
cv2.imshow("Bilinear", bilinear_img)
cv2.imshow("Bicubic", bicubic_img)

print("\nTekan tombol apa saja untuk menutup semua window...")
cv2.waitKey(0)
cv2.destroyAllWindows()