import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt
import time

# =========================
# LOAD IMAGE
# =========================
def load_images():
    img1 = cv2.imread('natural.jpeg', 0)
    img2 = cv2.imread('noisy.jpeg', 0)

    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256,256))

    return img1, img2

# =========================
# FFT
# =========================
def fft_analysis(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude = np.log(1 + np.abs(fshift))
    phase = np.angle(fshift)

    return fshift, magnitude, phase

# =========================
# FILTER
# =========================
def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    center = (rows//2, cols//2)

    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-center[0])**2 + (j-center[1])**2) <= cutoff:
                mask[i,j] = 1
    return mask

def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    mask = np.zeros((rows, cols))
    center = (rows//2, cols//2)

    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i-center[0])**2 + (j-center[1])**2)
            mask[i,j] = np.exp(-(d**2)/(2*(cutoff**2)))
    return mask

def apply_filter(img, mask):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(img_back)

# =========================
# NOTCH FILTER
# =========================
def notch_filter(shape, points, radius=10):
    mask = np.ones(shape)
    for (x,y) in points:
        cv2.circle(mask, (x,y), radius, 0, -1)
    return mask

# =========================
# WAVELET
# =========================
def wavelet_decompose(img):
    coeffs = pywt.wavedec2(img, 'haar', level=1)
    cA, (cH, cV, cD) = coeffs
    return cA, cH, cV, cD, coeffs

def wavelet_reconstruct(coeffs):
    return pywt.waverec2(coeffs, 'haar')

# =========================
# PSNR
# =========================
def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =========================
# MAIN
# =========================
img1, img2 = load_images()

# FFT
f1, mag1, phase1 = fft_analysis(img1)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img1, cmap='gray'); plt.title('Original')
plt.subplot(1,3,2); plt.imshow(mag1, cmap='gray'); plt.title('Magnitude')
plt.subplot(1,3,3); plt.imshow(phase1, cmap='gray'); plt.title('Phase')
plt.show()

# FILTERING
start = time.time()
mask = ideal_lowpass(img1.shape, 30)
filtered = apply_filter(img1, mask)
time_lowpass = time.time() - start

start = time.time()
gmask = gaussian_lowpass(img1.shape, 30)
gfiltered = apply_filter(img1, gmask)
time_gaussian = time.time() - start

start = time.time()
notch = notch_filter(img2.shape, [(100,100),(150,150)])
notch_result = apply_filter(img2, notch)
time_notch = time.time() - start

# tampilkan filtering
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(filtered, cmap='gray'); plt.title('Lowpass')
plt.subplot(1,3,2); plt.imshow(gfiltered, cmap='gray'); plt.title('Gaussian')
plt.subplot(1,3,3); plt.imshow(notch_result, cmap='gray'); plt.title('Notch')
plt.show()

# =========================
# WAVELET (UNTUK SS)
# =========================
cA, cH, cV, cD, coeffs = wavelet_decompose(img1)

plt.figure(figsize=(10,8))
plt.subplot(2,2,1); plt.imshow(cA, cmap='gray'); plt.title('LL (Approximation)')
plt.subplot(2,2,2); plt.imshow(cH, cmap='gray'); plt.title('LH (Horizontal)')
plt.subplot(2,2,3); plt.imshow(cV, cmap='gray'); plt.title('HL (Vertical)')
plt.subplot(2,2,4); plt.imshow(cD, cmap='gray'); plt.title('HH (Diagonal)')
plt.show()

# rekonstruksi wavelet
start = time.time()
recon_wavelet = wavelet_reconstruct(coeffs)
time_wavelet = time.time() - start

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(img1, cmap='gray'); plt.title('Original')
plt.subplot(1,2,2); plt.imshow(recon_wavelet, cmap='gray'); plt.title('Wavelet Recon')
plt.show()

# =========================
# PERBANDINGAN METRIK
# =========================
print("\nPERBANDINGAN METRIK")
print("-"*50)
print(f"{'Metode':<15} {'PSNR':<10} {'Waktu'}")
print("-"*50)

print(f"{'Lowpass':<15} {psnr(img1, filtered):<10.2f} {time_lowpass:.4f}")
print(f"{'Gaussian':<15} {psnr(img1, gfiltered):<10.2f} {time_gaussian:.4f}")
print(f"{'Notch':<15} {psnr(img1, notch_result):<10.2f} {time_notch:.4f}")
print(f"{'Wavelet':<15} {psnr(img1, recon_wavelet):<10.2f} {time_wavelet:.4f}")