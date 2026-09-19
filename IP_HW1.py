import numpy as np 
import cv2
from tqdm import tqdm

def nearest_neighbor(src,x,y):
    i=int(round(x))
    j=int(round(y))
    
    h,w=src.shape[:2]
    
    if 0<=i<h and 0<=j<w:
        return src[i,j]
    
    return np.array([0,0,0],dtype=np.uint8)

def bilinear(src,x,y):
    i=int(np.floor(x))
    j=int(np.floor(y))
    dx=x-i
    dy=y-j
    
    h,w=src.shape[:2]
    
    if i<0 or i>=h-1 or j<0 or j>=w-1:
        return np.array([0,0,0],dtype=np.uint8)
    
    p00=src[i,j].astype(np.float32)
    p01=src[i,j+1].astype(np.float32)
    p10=src[i+1,j].astype(np.float32)
    p11=src[i+1,j+1].astype(np.float32)
    
    top=(1-dx)*p00+dx*p01
    bottom=(1-dx)*p10+dx*p11
    
    result=(1-dy)*top+dy*bottom
    
    return np.clip(result,0,255).astype(np.uint8)

def cubic(p,x):
    a = (p[3] -3*p[2] +3*p[1] - p[0]) / 2
    b = (2*p[0] - 5*p[1] + 4*p[2] - p[3]) / 2
    c = (p[2] - p[0]) / 2
    d = p[1]
    
    return a*x**3 + b*x**2 + c*x + d

def bicubic(src,x,y):
    i=int(np.floor(x))
    j=int(np.floor(y))
    dx=x-i
    dy=y-j
    
    h,w=src.shape[:2]
    
    if i<0 or i>=h or j<0 or j>=w:
        return np.array([0,0,0],dtype=np.uint8)
    
    result=np.zeros(3,dtype=np.float32)
    
    for c in range(3):
        patch=np.zeros((4,4),dtype=np.float32)
        
        for m in range(4):
            for n in range(4):
                src_i=i-1+m
                src_j=j-1+n
                
                src_i=max(0,min(src_i,h-1))
                src_j=max(0,min(src_j,w-1))
                
                patch[m,n]=src[src_i,src_j,c]
                
        col_values=np.zeros(4,dtype=np.float32)
        for m in range(4):
            col_values[m]=cubic(patch[m,:],dx)
            
        result[c]=cubic(col_values,dy)
        
    return np.clip(result,0,255).astype(np.uint8)

def rotate(image,interpolation):
    h,w=image.shape[:2]
    center_y,center_x=h/2,w/2
    
    angle=np.radians(30)
    cos_a=np.cos(angle)
    sin_a=np.sin(angle)
    
    if interpolation=='nn':
        interpolate=nearest_neighbor
    elif interpolation=='bl':
        interpolate=bilinear
    elif interpolation=='bc':
        interpolate=bicubic
    else:
        print("NO")
    
    result=np.zeros_like(image)
    
    for y in tqdm(range(h),desc=f"rotation-{interpolation}"):
        for x in range(w):
            x_centered=x-center_x
            y_centered=y-center_y
            
            src_x=cos_a*x_centered-sin_a*y_centered+center_x
            src_y=sin_a*x_centered+cos_a*y_centered+center_y
            
            result[y,x]=interpolate(image,src_y,src_x)
    
    return result

def compute_matrix(src_points,dst_points):
    A=[]
    b=[]
    
    for i in range(4):
        x,y=src_points[i]
        xp,yp=dst_points[i]
        
        A.append([x,y,1,0,0,0,-x*xp,-y*xp])
        A.append([0,0,0,x,y,1,-x*yp,-y*yp])
        b.extend([xp,yp])
    
    A=np.array(A,dtype=np.float32)
    b=np.array(b,dtype=np.float32)
    
    h,_,_,_=np.linalg.lstsq(A,b,rcond=None)
    
    H=np.vstack([h.reshape(8,1),[[1]]]).reshape(3,3)
    
    return H
    
def homography(sign,coffee,interpolation):
    h_s,w_s=sign.shape[:2]
    h_c,w_c=coffee.shape[:2]
    
    src_points=np.array([
        (0,0),
        (w_c-1,0),
        (w_c-1,h_c-1),
        (0,h_c-1)
    ],dtype=np.float32)
    
    dst_points=np.array([
        (757,474),
        (1474,644),
        (1474,1522),
        (757,1611)
    ],dtype=np.float32)
    
    H=compute_matrix(src_points,dst_points)
    
    H_inv=np.linalg.inv(H)
    
    if interpolation=='nn':
        interpolate=nearest_neighbor
    elif interpolation=='bl':
        interpolate=bilinear
    elif interpolation=='bc':
        interpolate=bicubic
    else:
        print("NO")
        
    result=sign.copy()
    
    for y in tqdm(range(h_s),desc=f"Homography-{interpolation}"):
        for x in range(w_s):
            src_homo=H_inv@np.array([x,y,1])
            src_x=src_homo[0]/src_homo[2]
            src_y=src_homo[1]/src_homo[2]
            
            if 0<=src_x<w_c and 0<=src_y<h_c:
                result[y,x]=interpolate(coffee,src_y,src_x)
                
    
    return result
    
    

def main():
    # Read image
    signboard=cv2.imread("SignBoard.png")
    coffee=cv2.imread("Coffee.png")
    
    print(signboard.shape[:2])
    print(coffee.shape[:2])
    
    # Image rotate
    rotated_nn=rotate(signboard,'nn')
    cv2.imwrite("rotated_nn.png",rotated_nn)
    
    rotated_bl=rotate(signboard,'bl')
    cv2.imwrite("rotated_bl.png",rotated_bl)
    
    rotated_bc=rotate(signboard,'bc')
    cv2.imwrite("rotated_bc.png",rotated_bc)
    
    # Homography Warping
    warp_nn=homography(signboard,coffee,'nn')
    cv2.imwrite("warp_nn.png",warp_nn)
    
    warp_bl=homography(signboard,coffee,'bl')
    cv2.imwrite("warp_bl.png",warp_bl)
    
    warp_bc=homography(signboard,coffee,'bc')
    cv2.imwrite("warp_bc.png",warp_bc)
    

if __name__=="__main__":
    main()