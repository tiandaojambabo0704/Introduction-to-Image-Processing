import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from tqdm import tqdm


def compute_hist(image):
    hist=np.zeros(256,dtype=np.float64)
    
    for i in image.flatten():
        hist[i]+=1
        
    hist/=image.size
    return hist

def c_cdf(hist):
    cdf=np.cumsum(hist)
    return cdf

def hist_eq(image):
    hist_origin=compute_hist(image)
    cdf=c_cdf(hist_origin)
    
    lut=np.round(255*cdf).astype(np.uint8)
    equalized=lut[image]
    
    hist_eq=compute_hist(equalized)
    
    return equalized,hist_origin,hist_eq
    

def q1():
    print("---Q1---")
    img=cv2.imread("Q1.jpg")
    
    eq_img,hist_before,hist_after=hist_eq(img)
    
    cv2.imwrite("Q1_equalized.jpg",eq_img)
    
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    fig.suptitle("Q1",fontsize=14,fontweight="bold")
    
    axes[0,0].imshow(img,cmap="gray",vmin=0,vmax=255)
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")
    
    axes[0,1].imshow(eq_img,cmap="gray",vmin=0,vmax=255)
    axes[0,1].set_title("Equalized")
    axes[0,1].axis("off")
    
    axes[1,0].bar(range(256),hist_before,color="blue",width=1)
    axes[1,0].set_title("Before eq Histogram")
    axes[1,0].set_xlabel("Gray")
    axes[1,0].set_ylabel("Probability")
    
    axes[1,1].bar(range(256),hist_after,color="red",width=1)
    axes[1,1].set_title("After eq Histogram")
    axes[1,1].set_xlabel("Gray")
    axes[1,1].set_ylabel("Probability")
    
    plt.tight_layout()
    plt.savefig("Q1_result.png",dpi=150,bbox_inches="tight")
    plt.close()
    
def hist_map(src,ref):
    hist_src=compute_hist(src)
    hist_ref=compute_hist(ref)
    
    cdf_src=c_cdf(hist_src)
    cdf_ref=c_cdf(hist_ref)
    
    lut=np.zeros(256,dtype=np.uint8)
    
    for s in range(256):
        diff=np.abs(cdf_ref-cdf_src[s])
        r=np.argmin(diff)
        lut[s]=r
    
    output=lut[src]
    
    return output,cdf_src, cdf_ref, lut
    
        
def q2():
    print("---Q2---")
    src=cv2.imread("Q2_src.jpg")
    ref=cv2.imread("Q2_ref.jpg")
    output, cdf_src, cdf_ref, lut = hist_map(src, ref)
    
    cv2.imwrite("Q2_output.jpg", output)
    
    ref_output = compute_hist(ref)
    
    hist_output = compute_hist(output)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Q2", fontsize=14, fontweight="bold")
    
    axes[0, 0].imshow(src, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("src")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(ref, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("ref")
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(output, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("output")
    axes[0, 2].axis("off")
    
    axes[1, 0].plot(cdf_src, color="blue", label="src CDF")
    axes[1, 0].plot(cdf_ref, color="red", label="ref CDF")
    axes[1, 0].set_title("CDF Comparison")
    axes[1, 0].set_xlabel("Gray")
    axes[1, 0].set_ylabel("CDF")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(range(256), lut, color="purple", linewidth=1)
    axes[1, 1].set_title("Mapping Function")
    axes[1, 1].set_xlabel("input")
    axes[1, 1].set_ylabel("output")
    axes[1, 1].grid(True, alpha=0.3)
    
    # axes[1, 1].bar(range(256), ref_output, color="purple", width=1)
    # axes[1, 1].set_title("Ref Histogram")
    # axes[1, 1].set_xlabel("Gray")
    # axes[1, 1].set_ylabel("Probability")
    
    axes[1, 2].bar(range(256), hist_output, color="green", width=1)
    axes[1, 2].set_title("Output Histogram")
    axes[1, 2].set_xlabel("Gray")
    axes[1, 2].set_ylabel("Probability")
    
    plt.tight_layout()
    plt.savefig("Q2_result.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Shape:{src.shape}")
    print(f"Shape:{ref.shape}")
    
def pad_image(image,pad):
    h,w,c=image.shape
    padded=np.zeros((h+2*pad,w+2*pad,c),dtype=image.dtype)
    
    padded[pad:pad+h,pad:pad+w,:]=image
    return padded

def mean_filter(image,k):
    pad=k//2
    padded=pad_image(image.astype(np.float64),pad)
    h,w,c=image.shape 
    output=np.zeros((h,w),dtype=np.float64)
    
    for i in tqdm(range(h),desc=f"Mean{k}*{k}",leave=True):
        for j in range(w):
            patch=padded[i:i+k,j:j+k]
            output[i,j]=patch.mean()
            
    return np.clip(output,0,255).astype(np.uint8)

def median_filter(image,k):
    pad=k//2
    padded=pad_image(image.astype(np.float64),pad)
    h,w,c=image.shape 
    output=np.zeros((h,w),dtype=np.float64)
    
    for i in tqdm(range(h),desc=f"Median{k}*{k}",leave=True):
        for j in range(w):
            patch=padded[i:i+k,j:j+k].flatten()
            output[i,j]=np.median(patch)
            
    return np.clip(output,0,255).astype(np.uint8)
    
def q3():
    print("---Q3---")
    img=cv2.imread("Q3.jpg")
    
    mean3=mean_filter(img,3)
    cv2.imwrite("Q3_mean3.jpg",mean3)
    
    mean5=mean_filter(img,5)
    cv2.imwrite("Q3_mean5.jpg",mean5)
    
    median3=median_filter(img,3)
    cv2.imwrite("Q3_median3.jpg",median3)
    
    median5=median_filter(img,5)
    cv2.imwrite("Q3_median5.jpg",median5)
    
    print(f"Shape:{img.shape}")

if __name__=="__main__":
    q1()
    q2()
    q3()