import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_my_image(image_path, sample_image_path=None):
    """Analyze your own image"""
    
    img = cv2.imread(image_path)
    if img is None:
        print("Gambar tidak ditemukan!")
        return
    
    print("=== ANALISIS CITRA PRIBADI ===")
    
    # 1. Dimensi dan resolusi
    height, width, channels = img.shape
    resolution = width * height
    print(f"Dimensi       : {width} x {height}")
    print(f"Channels      : {channels}")
    print(f"Resolusi      : {resolution} pixels")
    
    # 2. Aspect Ratio
    aspect_ratio = width / height
    print(f"Aspect Ratio  : {aspect_ratio:.2f}")
    
    # 3. Konversi ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Ukuran RGB    : {img.nbytes} bytes")
    print(f"Ukuran Gray   : {gray.nbytes} bytes")
    
    # 4. Statistik
    print("\nStatistik RGB:")
    print(f"Mean  : {np.mean(img):.2f}")
    print(f"Std   : {np.std(img):.2f}")
    print(f"Min   : {np.min(img)}")
    print(f"Max   : {np.max(img)}")
    
    # 5. Histogram RGB
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10,5))
    
    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=col)
    
    plt.title("Histogram RGB")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()
    
    # 6. Perbandingan dengan sample (opsional)
    if sample_image_path:
        sample = cv2.imread(sample_image_path)
        if sample is not None:
            print("\n=== PERBANDINGAN DENGAN SAMPLE ===")
            print(f"Resolusi Pribadi : {resolution}")
            print(f"Resolusi Sample  : {sample.shape[1]*sample.shape[0]}")

if __name__ == "__main__":
    analyze_my_image("celtim.jpg")
