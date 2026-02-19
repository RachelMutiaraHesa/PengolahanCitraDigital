# 24343110 - Rachel Mutiara Hesa

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    path = "ALASROBAN.jpeg"
    print("Memuat gambar...")

    if not os.path.exists(path):
        print("File tidak ditemukan!")
        return

    img = cv2.imread(path)
    if img is None:
        print("Gagal membaca gambar!")
        return

    print("Gambar berhasil dibaca.")

    # =============================
    # ANALISIS PARAMETER
    # =============================
    height, width, channels = img.shape
    bit_depth_per_channel = img.dtype.itemsize * 8
    total_bit_depth = bit_depth_per_channel * channels
    aspect_ratio = width / height
    memory_bytes = height * width * channels * (bit_depth_per_channel / 8)
    memory_mb = memory_bytes / (1024 * 1024)

    print("\n========== INFORMASI CITRA ==========")
    print(f"Shape                : {img.shape}")
    print(f"Resolusi             : {width} x {height} piksel")
    print(f"Jumlah Channel       : {channels}")
    print(f"Tipe Data            : {img.dtype}")
    print(f"Bit depth / channel  : {bit_depth_per_channel} bit")
    print(f"Total bit depth      : {total_bit_depth} bit")
    print(f"Aspect Ratio         : {round(aspect_ratio,2)}")
    print(f"Ukuran Memori        : {round(memory_mb,2)} MB")

    # Simulasi
    new_width, new_height = width*2, height*2
    new_bit_depth = total_bit_depth / 2
    new_memory_bytes = new_width * new_height * (new_bit_depth / 8)
    new_memory_mb = new_memory_bytes / (1024 * 1024)

    print("\n--- Simulasi: Resolusi 2x & Bit Depth Setengah ---")
    print(f"Resolusi Baru        : {new_width} x {new_height}")
    print(f"Total Bit Depth Baru : {new_bit_depth} bit")
    print(f"Ukuran Memori Baru   : {round(new_memory_mb,2)} MB")

    # =============================
    # REPRESENTASI
    # =============================
    print("\n========== REPRESENTASI CITRA ==========")
    print("\n5x5 Piksel Pertama:")
    print(img[:5, :5])

    flattened = img.flatten()
    print("\n25 Nilai Pertama (Flattened):")
    print(flattened[:25])
    print(f"\nTotal Elemen dalam Flatten: {len(flattened)}")

    # =============================
    # MANIPULASI
    # =============================
    crop = img[height//4:height//2, width//4:width//2]
    resize = cv2.resize(img, (width//2, height//2))
    rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =============================
    # VISUALISASI
    # =============================
    plt.figure(figsize=(12,8))

    images = [img, crop, resize, rotate, gray]
    titles = ["Original", "Cropping", "Resizing", "Rotasi 90°", "Grayscale"]

    for i in range(len(images)):
        plt.subplot(2,3,i+1)
        if i == 4:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
