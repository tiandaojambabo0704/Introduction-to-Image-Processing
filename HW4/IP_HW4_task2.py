import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def laplacian_freq(img, alpha=15.0):
    M, N = img.shape[:2]
    
    f = np.zeros((M, N, 3), dtype=np.complex128)
    for c in range(3):
        f[:,:,c] = np.fft.fft2(img[:,:,c])
        f[:,:,c] = np.fft.fftshift(f[:,:,c])
    
    spectrum_before = np.zeros((M, N, 3))
    for c in range(3):
        spectrum_before[:,:,c] = np.log(1 + np.abs(f[:,:,c]))
    spectrum_before = (spectrum_before / (spectrum_before.max() + 1e-8) * 255).astype(np.uint8)
    
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    U, V = np.meshgrid(v, u)
    
    H = -(U**2 + V**2)
    H = H / (np.max(np.abs(H)) + 1e-8)  
    
    H_sharpen = 1 + alpha * H
    
    g = np.zeros((M, N, 3), dtype=np.complex128)
    for c in range(3):
        g[:,:,c] = f[:,:,c] * H_sharpen
    
    spectrum_after = np.zeros((M, N, 3))
    for c in range(3):
        spectrum_after[:,:,c] = np.log(1 + np.abs(g[:,:,c]))
    spectrum_after = (spectrum_after / (spectrum_after.max() + 1e-8) * 255).astype(np.uint8)
    
    for c in range(3):
        g[:,:,c] = np.fft.ifftshift(g[:,:,c])
        g[:,:,c] = np.fft.ifft2(g[:,:,c])
    
    result = np.real(g)
    result = np.clip(result, 0, 255)
    
    laplacian_result = np.zeros((M, N, 3))
    for c in range(3):
        laplacian_result[:,:,c] = np.real(np.fft.ifft2(np.fft.ifftshift(f[:,:,c] * H)))
    
    laplacian_result = np.abs(laplacian_result)
    laplacian_result = np.clip(laplacian_result, 0, 255)
    if laplacian_result.max() > 0:
        laplacian_result = (laplacian_result / laplacian_result.max() * 255).astype(np.uint8)
    else:
        laplacian_result = laplacian_result.astype(np.uint8)
    
    return result.astype(np.uint8), laplacian_result, spectrum_before, spectrum_after

def process(image_path, alpha=15.0, resize_ratio=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if resize_ratio:
        h, w = img.shape[:2]
        new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
        img = cv2.resize(img, (new_w, new_h))
    
    img_float = img.astype(np.float32)
    
    sharpened, laplacian, spectrum_before, spectrum_after = laplacian_freq(img_float, alpha)
    
    return img, laplacian, sharpened, spectrum_before, spectrum_after

def display(original, laplacian, sharpened, spectrum_before, spectrum_after, save_path, title):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Edge (Frequency Domain)')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(sharpened)
    plt.title('Sharpened Result')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(spectrum_before)
    plt.title('Fourier Spectrum (Before Filtering)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(spectrum_after)
    plt.title('Fourier Spectrum (After Filtering)')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    M, N = original.shape[:2]
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    U, V = np.meshgrid(v, u)
    H = -(U**2 + V**2)
    H = H / (np.max(np.abs(H)) + 1e-8)
    H_sharpen_vis = 1 + 15 * H
    H_sharpen_vis = np.clip(H_sharpen_vis, 0, 1)
    plt.imshow(H_sharpen_vis, cmap='hot')
    plt.title('Filter H_sharpen (u,v)')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def process_images(image_paths, alpha=15.0, resize_ratio=None, save_dir='task2'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        print(f"\nProcessing: {img_path}")
        
        original, laplacian, sharpened, spectrum_before, spectrum_after = process(img_path, alpha, resize_ratio)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_comparison.png")
        
        display(original, laplacian, sharpened, spectrum_before, spectrum_after, save_path, 
                        f"Frequency Domain Laplacian (Alpha: {alpha})")
        
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.jpg"), 
                   cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_laplacian_freq.jpg"), laplacian)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_sharpened_freq.jpg"), 
                   cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_spectrum_before.jpg"), spectrum_before)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_spectrum_after.jpg"), spectrum_after)

image_paths = ['1.jpg', '2.jpg', '3.jpg']


process_images(
    image_paths=image_paths,
    alpha=1.0,
    resize_ratio=1.0,
    save_dir='task2'
)

print("\nTask 2 Done! Results saved in 'task2' folder.")