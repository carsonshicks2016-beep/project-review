"""Experimental image-only registration before geometric expansion tracking."""
import copy
import cv2
import numpy as np
from .vision_tracks import MaskedExpansion


class BackgroundExpansion(MaskedExpansion):
    version='eye-background-registered-v4'

    def __init__(self,bundle):
        super().__init__(bundle)
        cv2.setNumThreads(1)
        self.anchor=self.baseline.copy();self.anchor_valid=self.baseline_valid.copy()
        self.transforms=np.repeat(np.eye(3)[None],2,axis=0)
        self.previous_rgb=np.asarray(bundle['rgb']).copy();self.previous_valid=np.asarray(bundle['valid']).copy()

    def manifest(self):
        result=super().manifest();result.update(id=self.version,
            registration='Image-only Lucas-Kanade tracks, forward/backward consistency, RANSAC homography; own-body pixels excluded',
            minimum_tracks=12,minimum_inlier_fraction=.7,minimum_feature_span_fraction=.1,
            ransac_pixel_threshold=2.,maximum_roundtrip_pixels=1.5,
            ambiguous_frame_policy='Zero activity and reset reference/tracks; expose registration failure',
            opencv_version=cv2.__version__,opencv_threads=1,opencv_ransac_seed=0)
        return result

    def registration(self,old,new,old_valid,new_valid):
        old_gray=cv2.cvtColor(old,cv2.COLOR_RGB2GRAY);new_gray=cv2.cvtColor(new,cv2.COLOR_RGB2GRAY)
        usable=old_valid&new_valid
        # Tiny differences can be a static scene or sparse object edges. This
        # fallback is explicit, fixed and reported separately from flow fitting.
        changed=np.mean((np.abs(new_gray.astype(float)-old_gray)>2)[usable]) if usable.any() else 1
        if changed<=.005:return np.eye(3),{'valid':True,'method':'near_static','changed_fraction':float(changed)}
        mask=cv2.erode(old_valid.astype(np.uint8)*255,np.ones((7,7),np.uint8))
        points=cv2.goodFeaturesToTrack(old_gray,200,.01,8,mask=mask)
        if points is None:return None,{'valid':False,'reason':'no_features'}
        moved,status,_=cv2.calcOpticalFlowPyrLK(old_gray,new_gray,points,None,winSize=(21,21),maxLevel=3)
        back,status_back,_=cv2.calcOpticalFlowPyrLK(new_gray,old_gray,moved,None,winSize=(21,21),maxLevel=3)
        a=points[:,0];b=moved[:,0];roundtrip=np.linalg.norm(back[:,0]-a,axis=1)
        ij=np.rint(b[:,::-1]).astype(int);inside=(ij[:,0]>=0)&(ij[:,0]<new_valid.shape[0])&(ij[:,1]>=0)&(ij[:,1]<new_valid.shape[1])
        accepted=(status[:,0]>0)&(status_back[:,0]>0)&(roundtrip<=1.5)&inside
        indices=np.flatnonzero(accepted);indices=indices[new_valid[ij[indices,0],ij[indices,1]]]
        if len(indices)<12:return None,{'valid':False,'reason':'insufficient_tracks','tracks':len(indices)}
        cv2.setRNGSeed(0)
        transform,inliers=cv2.findHomography(a[indices],b[indices],cv2.RANSAC,2.,maxIters=2000,confidence=.995)
        if transform is None:return None,{'valid':False,'reason':'no_fit'}
        inliers=inliers[:,0].astype(bool);fraction=float(inliers.mean());selected=a[indices][inliers]
        span=float(np.prod(np.ptp(selected,axis=0))/old_valid.size) if len(selected) else 0
        if fraction<.7 or span<.1 or not np.isfinite(transform).all():
            return None,{'valid':False,'reason':'unreliable_fit','inlier_fraction':fraction,'span_fraction':span}
        return transform,{'valid':True,'method':'homography','tracks':len(indices),'inlier_fraction':fraction,'span_fraction':span}

    def advance(self,bundle,dt):
        lum,valid=self.unpack(bundle);rgb=np.asarray(bundle['rgb'])
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive image interval required')
        if lum.shape!=self.baseline.shape:raise ValueError('Eye dimensions changed')
        registration=[];h,w=lum.shape[1:]
        for side in range(2):
            transform,info=self.registration(self.previous_rgb[side],rgb[side],self.previous_valid[side],valid[side]);registration.append(info)
            if transform is None:
                self.anchor[side]=lum[side];self.anchor_valid[side]=valid[side];self.transforms[side]=np.eye(3);self.tracks[side]=[]
            else:self.transforms[side]=user@example.com[side]
            self.baseline[side]=cv2.warpPerspective(self.anchor[side],self.transforms[side],(w,h),flags=cv2.INTER_LINEAR)
            self.baseline_valid[side]=cv2.warpPerspective(self.anchor_valid[side].astype(np.uint8),self.transforms[side],(w,h),flags=cv2.INTER_NEAREST).astype(bool)
        features=super().advance(bundle,dt)
        for side,feature in enumerate(features):
            feature['registration']=registration[side]
            if not registration[side]['valid']:
                feature['lplc2_hz']=0.;feature['expansion_per_second']=0.
        self.previous_rgb=rgb.copy();self.previous_valid=valid.copy()
        return features

    def snapshot(self):
        result=super().snapshot();result.update(anchor=self.anchor.copy(),anchor_valid=self.anchor_valid.copy(),
            transforms=self.transforms.copy(),previous_rgb=self.previous_rgb.copy(),previous_valid=self.previous_valid.copy())
        return result

    def restore(self,state):
        super().restore(state)
        for key in ('anchor','anchor_valid','transforms','previous_rgb','previous_valid'):
            value=np.asarray(state[key]);expected=getattr(self,key)
            if value.shape!=expected.shape or value.dtype!=expected.dtype or not np.isfinite(value).all():raise ValueError('Invalid registered vision state')
            setattr(self,key,value.copy())
