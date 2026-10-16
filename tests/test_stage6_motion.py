import numpy as np
import pytest
from src.evaluation.stage6_motion import diagnose,history_features

def line(speed=1.,direction=(1.,0.)):
 d=np.asarray(direction);h=np.arange(-7,1)[:,None]*.4*speed*d;y=np.arange(1,13)[:,None]*.4*speed*d
 return h,y

def test_perfect_cv_units():
 h,y=line();r,s=diagnose(h,y,y)
 assert r['ADE_m']==0 and r['FDE_m']==0 and r['predicted_mean_speed_mps']==pytest.approx(1)
 assert r['history_heading_bin']=='E' and s[-1]['horizon_seconds']==pytest.approx(4.8)
 assert not any(v for k,v in r.items() if k.startswith('tag_') and k!='tag_sustained_future')

def test_endpoint_decomposition():
 h,y=line(direction=(.6,.8));p=y+np.array([1.,-2.]);r,_=diagnose(h,y,p)
 assert r['endpoint_along_error_m']**2+r['endpoint_cross_error_m']**2==pytest.approx(r['FDE_m']**2)

def test_translation_invariance():
 h,y=line();p=y*.7;offset=np.array([10.,-20.]);a,sa=diagnose(h,y,p);b,sb=diagnose(h+offset,y+offset,p+offset)
 for k in a:
  if isinstance(a[k],float):assert b[k]==pytest.approx(a[k],abs=1e-10)
  else:assert b[k]==a[k]

def test_zero_direction_is_missing():
 z=np.zeros((8,2));y=np.zeros((12,2));r,s=diagnose(z,y,y)
 assert r['history_heading_bin']=='unreliable' and r['endpoint_direction_error_deg'] is None
 assert r['endpoint_along_error_m'] is None and all(t['cross_error_m'] is None for t in s)

def test_sustained_motion_deceleration_tag():
 h,y=line();steps=np.linspace(1.,.1,12)*.4;p=np.stack([steps.cumsum(),np.zeros(12)],axis=1)
 r,_=diagnose(h,y,p);assert r['tag_sustained_but_predicted_deceleration']

def test_false_motion_and_distinction_from_history():
 h=np.zeros((8,2));_,p=line();r,_=diagnose(h,np.zeros((12,2)),p)
 assert r['tag_future_stationary_predicted_moving'] and r['tag_history_stationary_predicted_moving']
 r2,_=diagnose(h,p,p);assert not r2['tag_future_stationary_predicted_moving']

def test_direction_and_displacement_tags():
 h,y=line();p=np.stack([np.zeros(12),y[:,0]],axis=1);r,_=diagnose(h,y,p)
 assert r['endpoint_direction_error_deg']==pytest.approx(90) and r['tag_direction_error']
 assert diagnose(h,y,y*.1)[0]['tag_displacement_undershoot']
 assert diagnose(h,y,y*3)[0]['tag_displacement_overshoot']

def test_history_features_do_not_access_future():
 h,y=line();a=history_features(h);diagnose(h,y+20,y-30)
 assert a==history_features(h)

@pytest.mark.parametrize('bad',[np.full((8,2),np.nan),np.zeros((7,2)),np.full((8,2),np.inf)])
def test_invalid_history_rejected(bad):
 with pytest.raises(ValueError):history_features(bad)
