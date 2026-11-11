import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from mvs import *
from synthetic import scene

def test_projection():
    ims,cams,gt,box=scene(16);c=cams[3];X=np.array([[.02,.01,.03],[-.01,.02,-.03]])
    uv,z=project(X,**c);np.testing.assert_allclose(backproject(uv,z,**c),X,atol=1e-12)
def test_ncc():
    a=np.random.default_rng(0).random((4,25,3));np.testing.assert_allclose(ncc(a,2*a+3),1,atol=1e-12)
    assert np.isnan(ncc(a,np.ones_like(a))).all()
def test_votes():
    c,w=vote(np.array([[.9],[.8],[.2],[.1]]),.6);np.testing.assert_allclose(c,.85);np.testing.assert_allclose(w,.3125)
    assert np.isneginf(vote(np.array([[.9],[.6]]),.6)[0]).all()
def test_patch():
    yy,xx=np.mgrid[:8,:8];im=np.repeat((xx+2*yy)[:,:,None],3,axis=2).astype(float);p,ok=patches(im,np.array([[3.25,2.5]]),1)
    assert ok[0];np.testing.assert_allclose(p,8.25)
def test_fusion():
    ims,cams,gt,box=scene(32);v,f=fuse(gt,[np.isfinite(x).astype(float) for x in gt],cams,box,voxel=.006)
    assert len(f)>100
    used=v[np.unique(f)];err=abs(np.max(abs(used),axis=1)-.075)
    assert np.median(err)<.004

def test_bounds():
    c=dict(K=np.eye(3),R=np.eye(3),t=np.array([0.,0.,0.]));uv=np.array([[0.,0.],[3.,0.]])
    lo,hi=ray_bounds(uv,c,np.array([[-1,-1,2],[1,1,4]]))
    assert lo[0]==2 and hi[0]==4 and hi[1]<lo[1]

def test_single_neighbor_cannot_vote():
    assert np.isneginf(vote(np.array([[.99]]),.6)[0]).all()

def test_neighbor_separation():
    _,c,_,_=scene(16)
    for i in range(len(c)):
        ns=neighbors(c,i);assert i not in ns and len(ns)==4
        for j in ns:
            assert np.degrees(np.arccos(np.clip(c[i]['R'][2]@c[j]['R'][2],-1,1)))>=4
