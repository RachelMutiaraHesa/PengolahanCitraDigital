# ============================================
# PRAKTIKUM 3: TRANSFORMASI GEOMETRIK DAN INTERPOLASI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== PRAKTIKUM 3: TRANSFORMASI GEOMETRIK DAN INTERPOLASI ===\n")

# =====================================================
# FUNGSI MEMBUAT GAMBAR TEST
# =====================================================

def create_test_image(size=256):
    img = np.zeros((size, size), dtype=np.uint8)

    # Garis tengah
    cv2.line(img, (size//2, 0), (size//2, size), 200, 1)
    cv2.line(img, (0, size//2), (size, size//2), 200, 1)

    # Lingkaran
    cv2.circle(img, (size//2, size//2), size//8, 255, 2)

    # Persegi
    square_size = size//6
    cv2.rectangle(img,
                  (size//4 - square_size//2, size//4 - square_size//2),
                  (size//4 + square_size//2, size//4 + square_size//2),
                  150, 2)

    # Garis diagonal
    cv2.line(img, (0, 0), (size, size), 180, 1)
    cv2.line(img, (size, 0), (0, size), 180, 1)

    return img


# =====================================================
# FUNGSI TRANSFORMASI
# =====================================================

def apply_transformation(image, transformation_name, params=None):
    h, w = image.shape

    if transformation_name == 'translation':
        M = np.float32([[1, 0, 50], [0, 1, 30]])
        return cv2.warpAffine(image, M, (w, h))

    elif transformation_name == 'rotation':
        M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
        return cv2.warpAffine(image, M, (w, h))

    elif transformation_name == 'scaling':
        resized = cv2.resize(image, None, fx=1.5, fy=1.5)
        return resized[:h, :w]

    elif transformation_name == 'shearing':
        M = np.float32([[1, 0.3, 0], [0.2, 1, 0]])
        return cv2.warpAffine(image, M, (w, h))

    else:
        return image.copy()


# =====================================================
# FUNGSI PERBANDINGAN INTERPOLASI
# =====================================================

def compare_interpolation_methods(image, scale_factor=0.25):
    h, w = image.shape
    new_size = (int(w * scale_factor), int(h * scale_factor))

    methods = [
        ('Nearest Neighbor', cv2.INTER_NEAREST),
        ('Bilinear', cv2.INTER_LINEAR),
        ('Bicubic', cv2.INTER_CUBIC),
        ('Lanczos', cv2.INTER_LANCZOS4)
    ]

    results = []

    for name, flag in methods:
        down = cv2.resize(image, new_size, interpolation=flag)
        up = cv2.resize(down, (w, h), interpolation=flag)

        mse = np.mean((image.astype(float) - up.astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')

        results.append({
            'name': name,
            'downscaled': down,
            'upscaled': up,
            'mse': mse,
            'psnr': psnr
        })

    return results


# =====================================================
# 1. TRANSFORMASI DASAR
# =====================================================

print("1. TRANSFORMASI GEOMETRIK DASAR")

test_img = create_test_image(300)

transformations = [
    ('Original', 'none'),
    ('Translation', 'translation'),
    ('Rotation', 'rotation'),
    ('Scaling', 'scaling'),
    ('Shearing', 'shearing')
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for idx, (title, t_type) in enumerate(transformations):

    if t_type == 'none':
        result = test_img
    else:
        result = apply_transformation(test_img, t_type)

    axes[idx].imshow(result, cmap='gray')
    axes[idx].set_title(title)
    axes[idx].axis('off')

for i in range(len(transformations), len(axes)):
    axes[i].axis('off')

plt.suptitle("Transformasi Geometrik Dasar")
plt.tight_layout()
plt.show()


# =====================================================
# 2. PERBANDINGAN METODE INTERPOLASI (FIX ERROR)
# =====================================================

print("\n2. PERBANDINGAN METODE INTERPOLASI")

test_img_detail = create_test_image(400)

interpolation_results = compare_interpolation_methods(test_img_detail, 0.25)

fig, axes = plt.subplots(3, 5, figsize=(20, 12))

# Kolom 0: Original
axes[0, 0].imshow(test_img_detail, cmap='gray')
axes[0, 0].set_title("Original 400x400")
axes[0, 0].axis('off')

axes[1, 0].axis('off')
axes[2, 0].axis('off')

for idx, result in enumerate(interpolation_results):
    col = idx + 1

    # Downscaled
    axes[0, col].imshow(result['downscaled'], cmap='gray')
    axes[0, col].set_title(f"{result['name']}\nDownscaled")
    axes[0, col].axis('off')

    # Upscaled
    axes[1, col].imshow(result['upscaled'], cmap='gray')
    axes[1, col].set_title("Upscaled")
    axes[1, col].axis('off')

    # Error map
    error_map = np.abs(test_img_detail.astype(float) - result['upscaled'].astype(float))
    im = axes[2, col].imshow(error_map, cmap='hot')
    axes[2, col].set_title(f"MSE:{result['mse']:.1f}\nPSNR:{result['psnr']:.1f}")
    axes[2, col].axis('off')

    plt.colorbar(im, ax=axes[2, col])

plt.suptitle("Perbandingan Metode Interpolasi")
plt.tight_layout()
plt.show()

print("\n3. KOORDINAT HOMOGEN")

points = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
], dtype=np.float32)

# Tambahkan koordinat homogen
points_h = np.hstack([points, np.ones((4, 1))])

# Matriks Transformasi
T = np.array([[1, 0, 2],
              [0, 1, 1],
              [0, 0, 1]])

R = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],
              [np.sin(np.pi/4),  np.cos(np.pi/4), 0],
              [0, 0, 1]])

S = np.array([[2, 0, 0],
              [0, 1.5, 0],
              [0, 0, 1]])

transformed = (T @ R @ S @ points_h.T).T

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(points[:,0], points[:,1], label="Original", s=100)
ax.scatter(transformed[:,0], transformed[:,1], label="Transformed", s=100)

ax.set_xlim(-3,5)
ax.set_ylim(-3,5)
ax.grid(True)
ax.legend()
ax.set_title("Transformasi dengan Koordinat Homogen")
plt.show()
# =====================================================
# 4. ANALISIS DETAIL INTERPOLASI (ROTASI) - FIXED
# =====================================================

print("\n4. ANALISIS DETAIL INTERPOLASI")

pattern = np.zeros((100,100), dtype=np.uint8)

for i in range(100):
    for j in range(100):
        pattern[i,j] = (i+j) % 256

methods = [
    ('Nearest', cv2.INTER_NEAREST),
    ('Bilinear', cv2.INTER_LINEAR),
    ('Bicubic', cv2.INTER_CUBIC)
]

cols = len(methods) + 1

fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))

# ======================
# KOLOM 0 = ORIGINAL
# ======================
axes[0,0].imshow(pattern, cmap='gray')
axes[0,0].set_title("Original Pattern")
axes[0,0].axis('off')

axes[1,0].plot(pattern[50,:])
axes[1,0].set_title("Original Profile")
axes[1,0].set_ylim(0,255)
axes[1,0].grid(True)

# ======================
# METODE INTERPOLASI
# ======================
for idx, (name, flag) in enumerate(methods):
    col = idx + 1

    M = cv2.getRotationMatrix2D((50,50), 30, 1)
    rotated = cv2.warpAffine(pattern, M, (100,100), flags=flag)

    # Gambar hasil rotasi
    axes[0,col].imshow(rotated, cmap='gray')
    axes[0,col].set_title(name)
    axes[0,col].axis('off')

    # Profile intensitas
    axes[1,col].plot(rotated[50,:])
    axes[1,col].set_title("Intensity Profile")
    axes[1,col].set_ylim(0,255)
    axes[1,col].grid(True)

plt.suptitle("Analisis Detail: Efek Interpolasi pada Rotasi", fontsize=14)
plt.tight_layout()
plt.show()

print("\n5. AFFINE vs PERSPEKTIF")

grid = np.zeros((300,300), dtype=np.uint8)

for i in range(0,300,20):
    cv2.line(grid,(i,0),(i,300),200,1)
    cv2.line(grid,(0,i),(300,i),200,1)

# Affine
pts1 = np.float32([[50,50],[250,50],[50,250]])
pts2 = np.float32([[30,70],[220,30],[70,260]])
M_aff = cv2.getAffineTransform(pts1,pts2)
affine = cv2.warpAffine(grid,M_aff,(300,300))

# Perspective
pts1p = np.float32([[50,50],[250,50],[250,250],[50,250]])
pts2p = np.float32([[0,0],[300,0],[250,300],[50,300]])
M_p = cv2.getPerspectiveTransform(pts1p,pts2p)
persp = cv2.warpPerspective(grid,M_p,(300,300))

fig, axes = plt.subplots(1,3, figsize=(15,5))

axes[0].imshow(grid, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(affine, cmap='gray')
axes[1].set_title("Affine")
axes[1].axis('off')

axes[2].imshow(persp, cmap='gray')
axes[2].set_title("Perspective")
axes[2].axis('off')

plt.tight_layout()
plt.show()
print("\n6. APLIKASI PRAKTIS: IMAGE REGISTRATION DENGAN TRANSFORMASI GEOMETRIK")

def demonstrate_image_registration():
    """Demonstrate image registration using ECC algorithm"""
    
    ref_img = create_test_image(256)

    # Create transformed (moving) image
    M_true = cv2.getRotationMatrix2D((128, 128), 15, 0.9)
    M_true[0, 2] += 20
    M_true[1, 2] += 15

    moving_img = cv2.warpAffine(ref_img, M_true, (256, 256))

    # Add noise
    noise = np.random.normal(0, 10, moving_img.shape)
    moving_img = np.clip(moving_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Convert to float32 (required by ECC)
    ref_float = ref_img.astype(np.float32)
    moving_float = moving_img.astype(np.float32)

    # Initial warp matrix (identity)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_float,
            moving_float,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria
        )

        # Apply estimated transformation
        aligned_img = cv2.warpAffine(
            moving_img,
            warp_matrix,
            (256, 256),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

    except:
        print("Registration gagal. Menggunakan moving image sebagai fallback.")
        aligned_img = moving_img

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ref_img, cmap='gray')
    axes[0].set_title("Reference Image")
    axes[0].axis('off')

    axes[1].imshow(moving_img, cmap='gray')
    axes[1].set_title("Moving Image\n(Rotated + Translated + Noise)")
    axes[1].axis('off')

    axes[2].imshow(aligned_img, cmap='gray')
    axes[2].set_title("Aligned Image\n(After Registration)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("\nTrue Transformation Matrix:")
    print(np.round(M_true, 3))

    print("\nEstimated Transformation Matrix:")
    print(np.round(warp_matrix, 3))


# Jalankan demonstrasi
demonstrate_image_registration()