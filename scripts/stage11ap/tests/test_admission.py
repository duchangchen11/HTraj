import io, pickle, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scripts.stage11ap.adapters import SceneDatasetAdapter,scene_collate_adapter,iter_all_scenes
from scripts.stage11ap.safe_data import RestrictedDataUnpickler
from scripts.stage11ap.checkpoint_io import RestrictedCheckpointUnpickler
# archive script is also a CLI; add its isolated helper import path.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from archive_audit import validate_members

def data():
    return pd.DataFrame([dict(sceneId=s,metaId=m,x=float(t),y=float(m),frame=t) for s,m in [('a',0),('b',1),('b',2)] for t in range(20)])

def test_ragged_scene_adapter_preserves_values():
    df=data();ds=SceneDatasetAdapter(df,.4,20)
    assert len(ds)==2 and ds[0][0].shape==(1,20,2) and ds[1][0].shape==(2,20,2)
    np.testing.assert_array_equal(ds[1][0],df[df.sceneId=='b'][['x','y']].to_numpy().astype('float32').reshape(2,20,2)*.4)
    assert scene_collate_adapter([ds[1]])[0].shape==(2,20,2)

def test_no_silent_scene_batch_concatenation():
    ds=SceneDatasetAdapter(data(),1.,20)
    with pytest.raises(ValueError):scene_collate_adapter([ds[0],ds[1]])

def test_incomplete_trajectory_rejected():
    with pytest.raises(ValueError):SceneDatasetAdapter(data().iloc[:-1],1.,20)

def test_no_hidden_26_scene_epoch_cap():
    assert list(iter_all_scenes(range(50)))==list(range(50))

def test_restricted_pickle_does_not_execute_global():
    # A GLOBAL for eval is refused before any callable executes.
    with pytest.raises(pickle.UnpicklingError,match='Global blocked'):
        RestrictedDataUnpickler(io.BytesIO(b'cbuiltins\neval\n.')).load()

def test_plain_numpy_pickle_data_roundtrip():
    a=np.arange(6,dtype=np.float64).reshape(3,2)
    b=RestrictedDataUnpickler(io.BytesIO(pickle.dumps(a,protocol=4))).load()
    np.testing.assert_array_equal(a,b)

def test_checkpoint_unknown_global_blocked():
    with pytest.raises(pickle.UnpicklingError,match='Checkpoint global blocked'):
        RestrictedCheckpointUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()

def test_inference_interface_has_no_future_or_GT_parameter():
    import inspect
    from scripts.stage11ap.pipeline_admission import predicted_paths
    assert list(inspect.signature(predicted_paths).parameters)==['model','image','history','params','device']

def test_augmentation_shapes_ids_and_original_row_values(tmp_path):
    import cv2,importlib.util
    from scripts.stage11ap.adapters import augment_data_adapter
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('stage11ap_original_preprocessing',root/'experiments/stage11ap/cache/official/ynet/utils/preprocessing.py')
    orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
    p=tmp_path/'a';p.mkdir();cv2.imwrite(str(p/'reference.jpg'),np.ones((40,60,3),dtype=np.uint8)*100)
    df=data();df=df[df.sceneId=='a'].copy()
    aug,images=augment_data_adapter(df,tmp_path,orig)
    assert len(aug)==8*len(df) and len(images)==8
    pd.testing.assert_frame_equal(aug[aug.sceneId=='a'],df)
    assert images['a_rot90'].shape[:2]==(60,40)
    assert aug.metaId.nunique()==8 and aug.sceneId.nunique()==8

def test_recorded_real_pipeline_is_train_only_without_metrics():
    import json
    root=Path(__file__).resolve().parents[3]
    j=json.loads((root/'results/tables/stage11ap/full_pipeline_admission.json').read_text())
    assert j['passed'] and j['strict_load'] and j['checkpoint_keys']==776
    assert not j['metrics_computed'] and not j['loss_computed'] and j['optimizer_updates']==0
    assert j['train_only'] and not j['GT_provided_to_prediction']
    assert all(r['path_shape']==[20,1,12,2] for r in j['cases'])

@pytest.mark.parametrize('name',['../bad','/bad','a/../../b','C:\\bad','a\\b'])
def test_archive_path_traversal_rejected(name):
    with pytest.raises(ValueError):validate_members([zipfile.ZipInfo(name)],10<<30)

def test_archive_space_and_ratio_limits():
    z=zipfile.ZipInfo('data/train.pkl');z.file_size=100;z.compress_size=80
    assert validate_members([z],10<<30)==100
    with pytest.raises(ValueError):validate_members([z],100)
    z.file_size=2<<20;z.compress_size=1
    with pytest.raises(ValueError):validate_members([z],10<<30)
