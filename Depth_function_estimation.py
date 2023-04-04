# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:46:01 2023

@author: marti
"""


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv 


# %% Functions

### Homogeneous coordinates ### 
def hom(Y,s=1):
    # Return homogeneous coordinates, if scale S is not defined S=1
    return np.vstack((Y,np.ones((1,Y.shape[1]))*s)) 

### Inverse homogeneous coordinates ###  
def ihom(Y,s=1):
    # Return cartesian coordinates of Hs with scale S
    x = Y[:-1,:]
    if s != 0 :
       x = (s/Y[-1,0]) * x
    return x

### Mesh coordinates ###
def mesh(rows,cols):
    # coordinates 
    x  = np.linspace(0,cols-1,cols)
    y  = np.linspace(0,rows-1,rows)
    xx,yy = np.meshgrid(x,y)
    
    x = xx.flatten()
    y = yy.flatten()
    
    xy = np.matrix((x,y))
    return xy

### Cloud point create ###
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    
    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

# %% 
def depth_estimation(img,disp,b,KR,KL):
    """
    Depth estimation function
    Inputs: img - RGB Image (nxm)
            disp - disparity map (nxm)
            b - baseline distance 
            KR - Intrinsis parameters matrix (3x3) 
            KL - Intrinsic parameters matrix (3x3)
    Output: pts - Space coordinates (nxmx3)
    """
    # Image dimension 
    rows,cols = disp.shape[:2]
    # Mesh coordinates 
    pl = mesh(rows,cols)
    # Homogeneous coordinate of disparity 
    H0 = hom(np.matrix((disp.flatten())),0)      
    # Corresponding point     
    pr = (pl-H0)
    # identity matrix
    I  = np.identity(3)
    # Camera matrix
    Cl = KL*np.concatenate((I,np.zeros((3,1))),axis=1)
    Cr = KR*np.concatenate((I,-I*np.matrix((b,0,0)).T),axis=1)
    # Homogeneous coordinates of pl and pr 
    hpl = hom(pl)
    hpr = hom(pr)
    vz = np.zeros((3,1))
    # Estimates linear homogeneous method 
    pt = np.zeros((3,cols*rows))
    for i in range(cols*rows):
        a1 = np.concatenate((Cl,hpl[:,i],vz),axis=1)
        a2 = np.concatenate((Cr,vz,hpr[:,i]),axis=1)
    
        A = np.concatenate((a1,a2),axis=0) 
        
        U,S,V = np.linalg.svd(A)
        
        V = np.transpose(V)
        
        p=ihom(V[:4,-1])
        
        pt[0,i] = p[0]
        pt[1,i] = p[1]
        pt[2,i] = p[2]

    # Coordinates points 
    pts = np.zeros([rows,cols,3])
    pts[:,:,0] = np.reshape(pt[0,:],(rows,cols))
    pts[:,:,1] = np.reshape(pt[1,:],(rows,cols))
    pts[:,:,2] = np.reshape(pt[2,:],(rows,cols))

    # Color points 
    colors = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Remove disparity values 0
    mask = disp > 15# disp.min()

    out_points = pts[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    return pts 

#Read Image 
img = cv.imread('view5.png',cv.IMREAD_COLOR)  
# dpr = np.matrix(Image.open('disp5.png'))//3
disp = np.load('DispL_NOGT.npy')+200
# dpr = np.load('dispR.npy')
# baseline in mm
b = 0.160 
# Camera instrinsisc parameters
KR = np.matrix([[3740 , 0,disp.shape[1]//2],
               [0, 3740 , disp.shape[0]//2],
               [0, 0, 1]])

KL = np.matrix([[3740 , 0,disp.shape[1]//2],
               [0, 3740 , disp.shape[0]//2],
               [0, 0, 1]])  
# KR = np.matrix([[863.829, 0.0, 290.23],
#                [0.0, 878.734, 109.20],
#                [0, 0, 1]])

# KL = np.matrix([[837.93, 0, 296.44],
#                [0, 841.69, 186.85],
#                [0, 0, 1]])  

pts = depth_estimation(img,disp,b,KR,KL)


