"""Audit tooling only: no prediction quality, no formal model training."""
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.stage11a.sdd_tools import (parse_annotation,resize_transform,
    transform_xy,sample_history,semantic_violations)

ROOT=Path(__file__).resolve().parents[3]


def annotation(tmp_path,content):
    p=tmp_path/'annotations.txt';p.write_text(content);return p


def test_headerless_first_row_and_bbox_center(tmp_path):
    p=annotation(tmp_path,'0 10 20 30 60 0 0 0 1 "Pedestrian"\n1 0 0 10 10 1 1 0 0 "Biker"\n')
    df=parse_annotation(p)
    assert len(df)==2 and df.iloc[0]['frame']==0
    assert df.iloc[0]['x']==20 and df.iloc[0]['y']==40
    assert df.iloc[0]['generated']==1 and df.iloc[1]['lost']==1


@pytest.mark.parametrize('row',[
    '1 10 0 5 20 0 0 0 0 "Pedestrian"\n',
    '1 0 0 5 20 0 2 0 0 "Pedestrian"\n',
    '1 0 0 5 20 0 0 0 0 "Aircraft"\n',
    '1 0.5 0 5 20 0 0 0 0 "Pedestrian"\n'])
def test_invalid_annotations_rejected(tmp_path,row):
    with pytest.raises(ValueError):parse_annotation(annotation(tmp_path,row))


def test_duplicate_id_frame_rejected(tmp_path):
    row='1 0 0 5 20 0 0 0 0 "Pedestrian"\n'
    with pytest.raises(ValueError):parse_annotation(annotation(tmp_path,row+row))


@pytest.mark.parametrize('dims',[(1424,1088,570,435),(1980,1093,792,437),(1417,2019,567,808)])
def test_per_video_resize_roundtrip(dims):
    m=resize_transform(*dims)
    xy=np.array([[0.,0.],[10.25,20.5],[dims[0]-1.,dims[1]-1.]])
    np.testing.assert_allclose(transform_xy(transform_xy(xy,m),np.linalg.inv(m)),xy,atol=1e-12)
    np.testing.assert_allclose(transform_xy(np.array([[dims[0],dims[1]]]),m),[dims[2:]])


def test_only_real_consecutive_rows_used_for_history(tmp_path):
    rows=''.join(f'1 {f} 0 {f+4} 4 {f} 0 0 0 "Pedestrian"\n' for f in range(100))
    df=parse_annotation(annotation(tmp_path,rows))
    actor,frames,xy=sample_history(df)
    assert actor==1 and frames.tolist()==list(range(0,85,12))
    assert len(xy)==8 and np.allclose(xy[:,0],frames+2)


def test_cannot_invent_history(tmp_path):
    p=annotation(tmp_path,'1 0 0 2 2 0 0 0 0 "Pedestrian"\n')
    with pytest.raises(ValueError):sample_history(parse_annotation(p))


def test_outside_semantics_separate_and_not_clipped():
    labels=np.zeros((4,4),dtype=int);labels[1,1]=2
    result=semantic_violations(np.array([[1.1,1.1],[-.1,1.],[4.,1.],[0.,0.]]),labels,{2})
    assert result['obstacle'].tolist()==[True,False,False,False]
    assert result['outside'].tolist()==[False,True,True,False]


def test_RGB_is_not_a_semantic_class_map():
    with pytest.raises(ValueError):semantic_violations(np.array([[0.,0.]]),np.zeros((4,4,3)),{2})


def test_recorded_six_mirror_cases_do_not_claim_official_equivalence():
    data=json.loads((ROOT/'results/tables/stage11a/alignment_results.json').read_text())
    assert data['real_cases']==6
    assert not data['exact_original_video_registration_proven']
    assert not data['semantic_maps_available']
    for case in data['cases']:
        assert not case['official_archive_identity_verified']
        assert case['inverse_roundtrip_max_px']<1e-9


def test_official_source_blobs_unchanged():
    import hashlib
    data=json.loads((ROOT/'results/tables/stage11a/source_download.json').read_text())
    for r in data['files']:
        if r['verified']:
            p=ROOT/'experiments/stage11a/cache/official'/r['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
