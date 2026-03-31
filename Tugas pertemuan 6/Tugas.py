import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

img = cv2.imread('KUCING.jpeg', 0)
img = cv2.resize(img, (256,256))

def motion_psf(length, angle):
    psf = np.zeros((length,length))
    center = length//2
    angle = np.deg2rad(angle)
    x1 = int(center - (length/2)*np.cos(angle))
    y1 = int(center - (length/2)*np.sin(angle))
    x2 = int(center + (length/2)*np.cos(angle))
    y2 = int(center + (length/2)*np.sin(angle))
    cv2.line(psf,(x1,y1),(x2,y2),1,1)
    return psf/np.sum(psf)

def add_motion_blur(img, psf):
    return cv2.filter2D(img.astype(float),-1,psf)

def add_gaussian_noise(img, sigma=20):
    noise = np.random.normal(0,sigma,img.shape)
    return np.clip(img+noise,0,255)

def add_sp_noise(img, prob=0.05):
    noisy = img.copy()
    num = int(prob*img.size/2)
    coords = [np.random.randint(0,i,num) for i in img.shape]
    noisy[coords[0],coords[1]] = 255
    coords = [np.random.randint(0,i,num) for i in img.shape]
    noisy[coords[0],coords[1]] = 0
    return noisy

def inverse_filter(img, psf, eps=1e-3):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    H[np.abs(H)<eps]=eps
    F = G/H
    return np.abs(np.fft.ifft2(F))

def wiener_filter(img, psf, K=0.01):
    G = np.fft.fft2(img)
    H = np.fft.fft2(psf, s=img.shape)
    Hc = np.conj(H)
    F = (Hc/(np.abs(H)**2 + K))*G
    return np.abs(np.fft.ifft2(F))

def richardson_lucy(img, psf, iter=20):
    img = img.astype(float)
    est = img.copy()
    psf_mirror = np.flip(psf)
    for i in range(iter):
        conv = cv2.filter2D(est,-1,psf)
        conv[conv==0]=1e-8
        ratio = img/conv
        est = est*cv2.filter2D(ratio,-1,psf_mirror)
    return np.clip(est,0,255)

def psnr(a,b):
    mse = np.mean((a-b)**2)
    if mse==0:
        return 100
    return 10*np.log10(255**2/mse)

def mse(a,b):
    return np.mean((a-b)**2)

def ssim(img1,img2):
    C1=(0.01*255)**2
    C2=(0.03*255)**2
    mu1=cv2.GaussianBlur(img1,(11,11),1.5)
    mu2=cv2.GaussianBlur(img2,(11,11),1.5)
    sigma1=cv2.GaussianBlur(img1**2,(11,11),1.5)-mu1**2
    sigma2=cv2.GaussianBlur(img2**2,(11,11),1.5)-mu2**2
    sigma12=cv2.GaussianBlur(img1*img2,(11,11),1.5)-mu1*mu2
    return np.mean(((2*mu1*mu2+C1)*(2*sigma12+C2))/((mu1**2+mu2**2+C1)*(sigma1+sigma2+C2)))

psf = motion_psf(15,30)

blur = add_motion_blur(img, psf)
g_noise = add_gaussian_noise(blur,20)
sp_noise = add_sp_noise(blur,0.05)

datasets = {
    "Motion Blur": blur,
    "Gaussian+Blur": g_noise,
    "SP+Blur": sp_noise
}

results = {}

for name,data in datasets.items():
    start=time.time()
    inv = inverse_filter(data,psf)
    t1=time.time()-start

    start=time.time()
    wie = wiener_filter(data,psf,0.01)
    t2=time.time()-start

    start=time.time()
    rl = richardson_lucy(data,psf,20)
    t3=time.time()-start

    results[name]={
        "Inverse":(inv,t1),
        "Wiener":(wie,t2),
        "RL":(rl,t3)
    }

plt.figure(figsize=(12,8))
i=1
for name,data in datasets.items():
    plt.subplot(3,4,i); plt.imshow(data,cmap='gray'); plt.title(name); plt.axis('off'); i+=1
    for method,(res,t) in results[name].items():
        plt.subplot(3,4,i); plt.imshow(res,cmap='gray')
        p=psnr(img,res)
        plt.title(f"{method}\nPSNR:{p:.2f}")
        plt.axis('off')
        i+=1
plt.tight_layout()
plt.show()

print("\nMETRIK PERBANDINGAN")
print("-"*70)
for name,data in results.items():
    print("\n",name)
    for method,(res,t) in data.items():
        print(method,
              "PSNR:",round(psnr(img,res),2),
              "MSE:",round(mse(img,res),2),
              "SSIM:",round(ssim(img,res),3),
              "Time:",round(t,4))