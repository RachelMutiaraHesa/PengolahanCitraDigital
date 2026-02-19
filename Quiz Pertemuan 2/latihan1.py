# ===============================
# IMPORT LIBRARY
# ===============================
import cv2
import numpy as np

# ===============================
# FUNGSI ANALISIS MODEL WARNA
# ===============================
def analyze_color_model_suitability(image, application):
    """
    Analyze which color model is best for specific application
    
    Parameters:
    image: Input image (BGR format)
    application: 'skin_detection', 'shadow_removal', 
                 'text_extraction', 'object_detection'
    
    Returns:
    Dictionary containing best color model and analysis
    """

    # Validasi input gambar
    if image is None:
        raise ValueError("Input image is empty or failed to load.")

    analysis = {}

    if application == 'skin_detection':
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        analysis['best_model'] = 'YCrCb / HSV'
        analysis['reason'] = (
            "YCrCb memisahkan luminance (Y) dari chrominance (Cr, Cb) "
            "sehingga warna kulit lebih stabil terhadap perubahan pencahayaan. "
            "HSV juga efektif karena Hue merepresentasikan warna secara independen."
        )
        analysis['used_channels'] = ['Cr, Cb', 'Hue']
        analysis['example'] = 'Face detection, beauty camera, driver monitoring'

    elif application == 'shadow_removal':
        cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        analysis['best_model'] = 'LAB'
        analysis['reason'] = (
            "LAB memisahkan Lightness (L) dari informasi warna (a, b). "
            "Bayangan terutama mempengaruhi channel L sehingga dapat dikoreksi "
            "tanpa merusak warna asli."
        )
        analysis['used_channels'] = ['L channel']
        analysis['example'] = 'Citra jalan raya dan citra satelit'

    elif application == 'text_extraction':
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        analysis['best_model'] = 'Grayscale'
        analysis['reason'] = (
            "Ekstraksi teks bergantung pada perbedaan intensitas, "
            "bukan warna. Grayscale menyederhanakan proses thresholding "
            "dan deteksi tepi."
        )
        analysis['used_channels'] = ['Intensity']
        analysis['example'] = 'OCR dokumen dan plat nomor'

    elif application == 'object_detection':
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        analysis['best_model'] = 'HSV'
        analysis['reason'] = (
            "HSV memudahkan segmentasi objek berbasis warna melalui channel Hue, "
            "dan lebih tahan terhadap perubahan pencahayaan dibanding RGB."
        )
        analysis['used_channels'] = ['Hue, Saturation']
        analysis['example'] = 'Object tracking, robot vision, animal detection'

    else:
        analysis['best_model'] = 'RGB'
        analysis['reason'] = 'Default model when application is unknown.'
        analysis['used_channels'] = ['R, G, B']
        analysis['example'] = '-'

    return analysis

# ===============================
# LOAD GAMBAR (kucing.jpg)
# ===============================
image = cv2.imread("kucing.jpg")

if image is None:
    print("Gambar kucing.jpg tidak ditemukan. Pastikan sudah di-upload.")
else:
    print("Gambar berhasil dimuat.")

    # ===============================
    # PANGGIL FUNGSI
    # ===============================
    result = analyze_color_model_suitability(
        image=image,
        application="object_detection"
    )

    # ===============================
    # TAMPILKAN OUTPUT
    # ===============================
    print("\n=== HASIL ANALISIS MODEL WARNA ===")
    print("Best Color Model :", result['best_model'])
    print("Reason           :", result['reason'])
    print("Used Channels    :", result['used_channels'])
    print("Example Case     :", result['example'])