"""Analytic textured cube renderer with first-hit visibility and exact camera-z."""
import numpy as np

def scene(size=64,texture=True):
    images=[];cams=[];truth=[];K=np.array([[75*size/64,0,(size-1)/2],[0,75*size/64,(size-1)/2],[0,0,1.]])
    yy,xx=np.mgrid[:size,:size];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    for angle in np.linspace(0,2*np.pi,12,endpoint=False):
        C=np.array([.35*np.cos(angle),.35*np.sin(angle),.17]);f=-C/np.linalg.norm(C)
        right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(f,right);R=np.array([right,down,f]);t=-R@C
        r=rays@R; near=np.full((size,size),-np.inf);far=np.full((size,size),np.inf)
        for ax in range(3):
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(-.075-C[ax])/r[:,:,ax];b=(.075-C[ax])/r[:,:,ax]
            near=np.maximum(near,np.minimum(a,b));far=np.minimum(far,np.maximum(a,b))
        valid=(far>=near)&(near>0);z=np.where(valid,near,np.nan);X=C+r*np.nan_to_num(z)[:,:,None]
        if texture:
            rgb=np.stack([.5+.22*np.sin(130*X[:,:,0]+31*X[:,:,2])+.2*np.cos(105*X[:,:,1]),.5+.25*np.sin(119*X[:,:,1]+43*X[:,:,0]),.5+.24*np.cos(113*X[:,:,2]+61*X[:,:,0]+41*X[:,:,1])],-1)
        else:rgb=np.full((size,size,3),.7)
        rgb[~valid]=.05;images.append(rgb.clip(0,1));truth.append(z);cams.append(dict(K=K.copy(),R=R,t=t))
    return images,cams,truth,np.array([[-.09]*3,[.09]*3])
