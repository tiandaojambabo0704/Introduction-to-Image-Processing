import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def convolution(img, kernel, padding='zero'):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    
    if padding == 'zero':
        padded = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode='constant', constant_values=0)
    else:
        padded = np.pad(img, ((ph, ph), (pw, pw), (0, 0)), mode='edge')
    
    h, w = img.shape[:2]
    output = np.zeros_like(img)
    
    for i in tqdm(range(h), desc="Convolution"):
        for j in range(w):
            for c in range(3):
                region = padded[i:i+kh, j:j+kw, c]
                output[i, j, c] = np.sum(region * kernel)
    
    return output

def sharpen(image_path, kernel_type='8', alpha=1.0, resize_ratio=None, padding='zero'):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if resize_ratio:
        h, w = img.shape[:2]
        new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
        img = cv2.resize(img, (new_w, new_h))
    
    img_float = img.astype(np.float32)
    
    if kernel_type == '4':
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    else:
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    
    laplacian = convolution(img_float, kernel, padding)
    
    sharpened = img_float - alpha * laplacian
    sharpened = np.clip(sharpened, 0, 255)
    
    laplacian_vis = np.abs(laplacian)
    laplacian_vis = np.clip(laplacian_vis, 0, 255)
    laplacian_vis = (laplacian_vis / (laplacian_vis.max() + 1e-8) * 255).astype(np.uint8)
    
    return img, laplacian_vis, sharpened.astype(np.uint8)

def display(original, laplacian, sharpened, save_path, title):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Edge')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sharpened)
    plt.title('Sharpened Result')
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def process_images(image_paths, kernel_type='8', alpha=1.0, resize_ratio=None, padding='zero', save_dir='results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        original, laplacian, sharpened = sharpen(
            img_path, kernel_type, alpha, resize_ratio, padding
        )
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_sharpened.png")
        
        display(original, laplacian, sharpened, save_path, 
                        f"Kernel: {kernel_type}-neighbor, Alpha: {alpha}")
        
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_original.jpg"), 
                   cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_laplacian.jpg"), laplacian)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_sharpened.jpg"), 
                   cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))

image_paths = ['1.jpg','2.jpg','3.jpg']

process_images(
    image_paths=image_paths,
    kernel_type='8', # '4' '8;'
    alpha=1.0,
    resize_ratio=1.0,
    padding='zero', # 'zero' 'replicate'
    save_dir='task1'
)

print("\nDone!")