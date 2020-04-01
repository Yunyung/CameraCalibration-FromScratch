import numpy as np

def solve_homography(P, m):
    """
    Solve all homography matrix of each image 
    
    Args:
        P: objp, objpoint of image
        m: homographies[i], homography of image i

        
    Returns:
        H: Homography matrix of each image
    """
    
    A = []  
    for r in range(len(P)): 
        A.append([-P[r,0], -P[r,1], -1, 0, 0, 0, P[r,0]*m[r,0][0], P[r,1]*m[r,0][0], m[r,0][0]])
        A.append([0, 0, 0, -P[r,0], -P[r,1], -1, P[r,0]*m[r,0][1], P[r,1]*m[r,0][1], m[r,0][1]])
        
    u, s, vt = np.linalg.svd(A) # Solve s ystem of linear equations Ah = 0 using SVD
    # pick H from vt with the smallest s[i]
    H = np.reshape(vt[-1], (3,3))
    # normalization, let H[2,2] equals to 1
    H = (1/H.item(-1)) * H

    return H

def v_pq(p, q, H):
    v = np.array([
            H[0, p]*H[0, q],
            H[0, p]*H[1, q] + H[1, p]*H[0, q], 
            H[1, p]*H[1, q],
            H[2, p]*H[0, q] + H[0, p]*H[2, q],
            H[2, p]*H[1, q] + H[1, p]*H[2, q],
            H[2, p]*H[2, q]
        ])
    return v

def solve_intrinsic(homographies):
    """
    Use homographies to find out the intrinsic matrix K  
    
    Return: 
        K: intrisic matrix
        B: B = K^(-T) * K^(-1) 
    """

    v = []
    for i in range(len(homographies)):
        h = homographies[i]
        v.append(v_pq(0, 1, h))
        v.append(np.subtract(v_pq(0, 0, h), v_pq(1, 1, h)))
    
    v = np.array(v)
    u, s, vh = np.linalg.svd(v)
    b = vh[-1]

    # Make sure that B is positive definite
    if(b[0] < 0 or b[2] < 0 or b[5] < 0):
        b = -b
   
    B = [[b[0], b[1], b[3]],
         [b[1], b[2], b[4]],
         [b[3], b[4], b[5]]]
    B = np.array(B)
  
    # Solve K by cholesky factoriztion where B=K^(-T)K^(-1), L=K^(-T)
    L = np.linalg.cholesky(B)
    K = np.linalg.inv(L).transpose() * L[2,2]
    
    return K, B

def solve_extrinsics(intrinsic, homographies):
    """
    Find out the extrinsics of each image by intrinsic and homographyies
    
    Return:
        extrinsics: the extrensics matrix of each image
    """

    extrinsics = []
    inv_intrinsic = np.linalg.inv(intrinsic)
    for i in range(len(homographies)): # Iteratively find out the extrinsics of each image
        #h1, h2, h3 are columns in H
        h1 = homographies[i][:,0]
        h2 = homographies[i][:,1]
        h3 = homographies[i][:,2]
        
        lamda1 = 1/np.linalg.norm(np.dot(inv_intrinsic, h1))
        lamda2 = 1/np.linalg.norm(np.dot(inv_intrinsic, h2))
        lamda3 = (lamda1 + lamda2) / 2
        
        r1 = lamda1 * np.dot(inv_intrinsic, h1)
        r2 = lamda2 * np.dot(inv_intrinsic, h2)
        r3 = np.cross(r1, r2)
        t = lamda3 * np.dot(inv_intrinsic, h3)
        
        RT = np.array([r1, r2, r3, t]).transpose()
        extrinsics.append(RT)
        
    extrinsics = np.array(extrinsics)    
    return extrinsics  
