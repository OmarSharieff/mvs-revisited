"""Conservative RGB patch stereo and explicitly approximate projective fusion.
World-to-camera: Xc = R X + t. Depth is camera z; distances are meters.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

def project(X,K,R,t):
    q=X@R.T+t; h=q@K.T
    return h[...,:2]/h[...,2:3],q[...,2]

def backproject(uv,z,K,R,t):
    rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T
    return (rays*z[:,None]-t)@R

def neighbors(cams,i,k=4):
    axes=np.array([c['R'][2] for c in cams]); a=np.degrees(np.arccos(np.clip(axes@axes.T,-1,1)))
    out=[]
    for j in np.argsort(a[i]):
        if j!=i and a[i,j]>=4 and all(a[j,q]>=4 for q in out): out.append(int(j))
        if len(out)==k: break
    return out

def patches(im,uv,m):
    off=np.arange(m)-m//2; yy,xx=np.meshgrid(off,off,indexing='ij')
    x=uv[:,0,None]+xx.ravel(); y=uv[:,1,None]+yy.ravel()
    valid=(x.min(1)>=0)&(y.min(1)>=0)&(x.max(1)<im.shape[1]-1)&(y.max(1)<im.shape[0]-1)&np.isfinite(uv).all(1)
    x=np.nan_to_num(x);y=np.nan_to_num(y)
    p=np.stack([map_coordinates(im[:,:,c],[y,x],order=1,mode='nearest') for c in range(3)],-1)
    return p,valid

def ncc(a,b):
    a=a-a.mean(-2,keepdims=True);b=b-b.mean(-2,keepdims=True)
    ea=(a*a).sum((-2,-1)); eb=(b*b).sum((-2,-1)); den=np.sqrt(ea*eb)
    return np.divide((a*b).sum((-2,-1)),den,out=np.full_like(den,np.nan),where=(ea>1e-10)&(eb>1e-10)).clip(-1,1)

def vote(scores,threshold):
    ok=np.isfinite(scores)&(scores>threshold);count=ok.sum(0)
    corr=np.divide(np.where(ok,scores,0).sum(0),count,out=np.full(count.shape,-np.inf),where=count>=2)
    conf=np.where(ok,scores-threshold,0).sum(0)/(len(scores)*(1-threshold))
    return corr,conf

def ray_bounds(uv,cam,box):
    o=-cam['R'].T@cam['t']; r=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(cam['K']).T@cam['R']
    lo=np.zeros(len(uv));hi=np.full(len(uv),np.inf)
    for ax in range(3):
        parallel=abs(r[:,ax])<1e-12
        a=np.divide(box[0,ax]-o[ax],r[:,ax],out=np.full(len(uv),-np.inf),where=~parallel)
        b=np.divide(box[1,ax]-o[ax],r[:,ax],out=np.full(len(uv),np.inf),where=~parallel)
        lo=np.maximum(lo,np.minimum(a,b));hi=np.minimum(hi,np.maximum(a,b))
        if not box[0,ax]<=o[ax]<=box[1,ax]: hi[parallel]=-1
    return lo,hi

def depth(images,cams,i,box,m=5,threshold=.6,step=.005):
    c=cams[i];h,w=images[i].shape[:2];yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()].astype(float)
    ref,valid=patches(images[i],uv,m);lo,hi=ray_bounds(uv,c,box);valid &= hi>lo
    ids=np.where(valid)[0];out=np.full(h*w,np.nan);confidence=np.zeros(h*w);correlation=np.full(h*w,np.nan)
    ns=neighbors(cams,i)
    if len(ns)<2:return out.reshape(h,w),confidence.reshape(h,w),correlation.reshape(h,w)
    for start in range(0,len(ids),512):
        ix=ids[start:start+512]; p=uv[ix];best=np.full(len(ix),-np.inf);bestz=np.full(len(ix),np.nan);bc=np.zeros(len(ix))
        def evaluate(z):
            X=backproject(p,z,c['K'],c['R'],c['t']);sc=[]
            for j in ns:
                q,zj=project(X,**cams[j]);pat,ok=patches(images[j],q,m);s=ncc(ref[ix],pat);s[~ok|(zj<=0)]=np.nan;sc.append(s)
            corr,conf=vote(np.array(sc),threshold);corr[(z<lo[ix])|(z>hi[ix])]=-np.inf
            return corr,conf
        for off in np.arange(0,np.max(hi[ix]-lo[ix])+step*.5,step):
            z=lo[ix]+off;score,cf=evaluate(z);take=score>best;best[take]=score[take];bestz[take]=z[take];bc[take]=cf[take]
        coarse=bestz.copy()
        for off in np.linspace(-step,step,21):
            z=np.where(np.isfinite(coarse),coarse+off,lo[ix]);score,cf=evaluate(z);take=(score>best)&np.isfinite(coarse)
            best[take]=score[take];bestz[take]=z[take];bc[take]=cf[take]
        out[ix]=bestz;confidence[ix]=bc;correlation[ix]=np.where(np.isfinite(best),best,np.nan)
    return out.reshape(h,w),confidence.reshape(h,w),correlation.reshape(h,w)

def fuse(depths,confs,cams,box,voxel=.004,uniform=False):
    """Narrow-band projective fusion; not VripPack range-mesh scan conversion."""
    axes=[np.arange(box[0,a],box[1,a]+voxel*.5,voxel) for a in range(3)];shape=tuple(map(len,axes))
    X=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3);S=np.zeros(len(X));W=S.copy();band=4*voxel
    for d,cf,c in zip(depths,confs,cams):
        uv,z=project(X,**c);u=np.rint(uv[:,0]).astype(int);v=np.rint(uv[:,1]).astype(int)
        ok=(z>0)&(u>=0)&(u<d.shape[1])&(v>=0)&(v<d.shape[0]);idx=np.where(ok)[0]
        measured=d[v[idx],u[idx]];weight=cf[v[idx],u[idx]];res=measured-z[idx]
        keep=np.isfinite(res)&(abs(res)<=band)&(weight>0);idx=idx[keep];weight=weight[keep]
        if uniform:weight=np.ones_like(weight)
        S[idx]+=weight*res[keep]/band;W[idx]+=weight
    F=np.divide(S,W,out=np.ones_like(S),where=W>0).reshape(shape);observed=(W>0).reshape(shape)
    if not (F.min()<0<F.max()):return np.empty((0,3)),np.empty((0,3),int)
    verts,faces,_,_=marching_cubes(F,level=0,spacing=(voxel,)*3)
    # Keep triangles only when their containing grid cell has all eight corners observed.
    cent=verts[faces].mean(1)/voxel;cell=np.floor(cent).astype(int);cell=np.minimum(cell,np.array(shape)-2)
    keep=np.ones(len(faces),bool)
    for a in (0,1):
      for b in (0,1):
       for c in (0,1):keep &= observed[cell[:,0]+a,cell[:,1]+b,cell[:,2]+c]
    return verts+box[0],faces[keep]

def save_ply(path,vertices,faces):
    with open(path,'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        np.savetxt(f,vertices,fmt='%.8f');np.savetxt(f,np.c_[np.full(len(faces),3),faces],fmt='%d')
