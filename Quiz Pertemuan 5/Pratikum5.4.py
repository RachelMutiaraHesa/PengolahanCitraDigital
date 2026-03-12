import numpy as np
import cv2
import matplotlib.pyplot as plt

def adaptive_hybrid_filters():
    """
    Demonstrasi adaptive dan hybrid filtering
    """
    print("\nPRAKTIKUM 5.4: ADAPTIVE DAN HYBRID FILTERING")
    print("=" * 50)

    # Membuat citra dasar
    clean_img = np.zeros((256, 256), dtype=np.uint8)

    for i in range(0, 256, 32):
        cv2.rectangle(clean_img, (i, i), (i+16, i+16), 200, -1)

    # Tambahkan noise campuran
    noisy_img = clean_img.copy().astype(float)

    gaussian_noise = np.random.normal(0, 20, clean_img.shape)
    noisy_img += gaussian_noise

    salt_pepper = np.random.random(clean_img.shape)
    noisy_img[salt_pepper < 0.02] = 255
    noisy_img[salt_pepper > 0.98] = 0

    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    # Adaptive Mean Filter
    def adaptive_mean_filter(image, window_size=3, threshold=30):
        h, w = image.shape
        output = image.copy().astype(float)

        pad = window_size // 2
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        for i in range(h):
            for j in range(w):
                window = padded[i:i+window_size, j:j+window_size]
                local_mean = np.mean(window)
                local_var = np.var(window)

                if local_var > threshold:
                    output[i, j] = local_mean

        return np.clip(output, 0, 255).astype(np.uint8)

    # Hybrid filter
    def hybrid_filter(image):
        median = cv2.medianBlur(image, 3)
        gaussian = cv2.GaussianBlur(median, (3, 3), 0)
        return gaussian

    # Bilateral filter
    def bilateral_filter_custom(image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    filters = {
        'Noisy Image': lambda x: x,
        'Mean 3x3': lambda x: cv2.blur(x, (3,3)),
        'Median 3x3': lambda x: cv2.medianBlur(x,3),
        'Adaptive Mean': lambda x: adaptive_mean_filter(x),
        'Hybrid Filter': lambda x: hybrid_filter(x),
        'Bilateral Filter': lambda x: bilateral_filter_custom(x)
    }

    results = []

    fig, axes = plt.subplots(2,3, figsize=(15,10))
    axes = axes.ravel()

    for idx,(name,func) in enumerate(filters.items()):
        filtered = func(noisy_img)

        mse = np.mean((clean_img.astype(float)-filtered.astype(float))**2)
        psnr = 10*np.log10(255**2/mse) if mse>0 else float('inf')

        # SSIM sederhana
        def calculate_ssim(img1,img2):
            C1=(0.01*255)**2
            C2=(0.03*255)**2

            mu1=cv2.GaussianBlur(img1.astype(float),(11,11),1.5)
            mu2=cv2.GaussianBlur(img2.astype(float),(11,11),1.5)

            mu1_sq=mu1**2
            mu2_sq=mu2**2
            mu1_mu2=mu1*mu2

            sigma1_sq=cv2.GaussianBlur(img1.astype(float)**2,(11,11),1.5)-mu1_sq
            sigma2_sq=cv2.GaussianBlur(img2.astype(float)**2,(11,11),1.5)-mu2_sq
            sigma12=cv2.GaussianBlur(img1.astype(float)*img2.astype(float),(11,11),1.5)-mu1_mu2

            ssim=((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
            return np.mean(ssim)

        ssim=calculate_ssim(clean_img,filtered)

        results.append({
            "filter":name,
            "mse":mse,
            "psnr":psnr,
            "ssim":ssim
        })

        axes[idx].imshow(filtered,cmap='gray')
        axes[idx].set_title(f"{name}\nPSNR:{psnr:.1f}\nSSIM:{ssim:.3f}")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    # Tabel hasil
    print("\nADVANCED FILTER PERFORMANCE")
    print("-"*70)
    print(f"{'Filter':<20}{'MSE':<12}{'PSNR':<12}{'SSIM':<10}")
    print("-"*70)

    for r in results:
        print(f"{r['filter']:<20}{r['mse']:<12.2f}{r['psnr']:<12.2f}{r['ssim']:<10.3f}")

    return results


advanced_filter_results = adaptive_hybrid_filters()