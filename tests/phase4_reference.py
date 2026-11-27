"""Pinned independent upstream reference; no external repository/runtime imports.

Copied only classify/fuse from joseph316/03_TaFVLA:
https://github.com/joseph316/03_TaFVLA/blob/master/script/stage4_short_windows.py
blob 64a2fa7b32902f7a13ea57a5bfcaeaa131564b81
https://github.com/joseph316/03_TaFVLA/blob/master/script/stage6_fusion.py
blob e80a5261a8b344bc03261abcc52b0de275a4b23e
Original upstream phase names and nominal "actual_span_seconds" are preserved.
Short-episode and input validation are tested against our implementation separately.
"""
import numpy as np

def classify(t,f,seconds,cfg):
    dt=float(np.median(np.diff(t)));intervals=int(round(seconds/dt));points=intervals+1
    left=intervals//2;right=intervals-left
    windows=np.lib.stride_tricks.sliding_window_view(f,points)
    x=np.arange(points)*dt;x-=x.mean();duration=intervals*dt
    slopes=windows@x/(x@x);changes=windows[:,-1]-windows[:,0]
    floor=cfg['range_fraction']*cfg['force_iqr'];noise=cfg['noise_sigma']
    change_thr=max(3*np.sqrt(2)*noise,floor)
    slope_thr=max(3*noise/np.sqrt(x@x),floor/duration)
    inner=np.full(len(windows),'Transition / Unknown',object)
    inner[(slopes>=slope_thr)&(changes>=change_thr)]='Loading'
    inner[(slopes<=-slope_thr)&(changes<=-change_thr)]='Unloading'
    inner[(abs(slopes)<=slope_thr*.5)&(abs(changes)<=change_thr*.5)]='Holding'
    labels=np.full(len(f),'Transition / Unknown',object);labels[left:len(f)-right]=inner
    cuts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if b-a<cfg['min_run_frames']:labels[a:b]='Transition / Unknown'
    return labels,{'requested_seconds':seconds,'points':points,'actual_span_seconds':duration,'left_frames':left,'right_frames':right,'center_offset_seconds':(right-left)*dt/2,'change_threshold':change_thr,'slope_threshold':slope_thr}

L='Loading';U='Unloading';H='Holding';X='Transition / Unknown'

def fuse(a,b):
    loading=(a==L)|(b==L);unloading=(a==U)|(b==U);conflict=loading&unloading
    result=np.full(len(a),X,object);result[loading&~unloading]=L;result[unloading&~loading]=U;result[(a==H)&(b==H)]=H
    return result,conflict

