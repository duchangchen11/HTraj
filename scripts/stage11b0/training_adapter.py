"""Original Y-Net loss/teacher forcing in an independent train-only runner.

MIT source formulas: Human-Path-Prediction/ynet/train.py at a345ada.
Only data traversal, logging and frozen segmentation mode are controlled here.
"""
from .common import *
from .runtime import original_modules,set_training
import torch,numpy as np,cv2
import torch.nn.functional as F

def make_templates(cfg,device):
    img=original_modules().image
    size=int(4200*cfg['resize'])
    return (torch.as_tensor(img.create_dist_mat(size),dtype=torch.float32,device=device),
            torch.as_tensor(img.create_gaussian_heatmap_template(size,cfg['kernlen'],cfg['nsig'],False),dtype=torch.float32,device=device))

def preprocess_scene(scene,cfg,device):
    path=RES/'data/SDD/train'/scene/'reference.jpg'
    im=cv2.imread(str(path));assert im is not None
    images={scene:im}
    util=original_modules().image
    util.resize(images,cfg['resize']);util.pad(images,2**len(cfg['encoder_channels']))
    util.preprocess_image_for_segmentation(images)
    return images[scene].unsqueeze(0).to(device)

def fixed_semantics(net,image):
    net.semantic_segmentation.eval()
    assert not any(p.requires_grad for p in net.semantic_segmentation.parameters())
    with torch.no_grad():out=net.segmentation(image)
    assert torch.isfinite(out).all()
    return out.detach()

def trajectory_maps(coords,H,W,template):
    """Author round+template slicing, including out-of-image points within padded grid."""
    assert coords.ndim==3 and coords.shape[-1]==2
    points=coords.detach().cpu().numpy().reshape(-1,2)
    patches=original_modules().image.get_patch(template,points,H,W)
    assert all(p.shape==(H,W) for p in patches), 'Unrepresentable patch: retained sample, fail instead of drop/clip'
    return torch.stack(patches).reshape(coords.shape[0],coords.shape[1],H,W)

def training_loss(net,trajectory,semantic,cfg,templates,return_logits=False):
    assert trajectory.shape[1:]==(20,2) and semantic.shape[0]==1
    set_training(net)
    distance,gaussian=templates;H,W=semantic.shape[-2:]
    observed=trajectory_maps(trajectory[:,:8],H,W,distance)
    truth=trajectory[:,8:]
    target=trajectory_maps(truth,H,W,gaussian)
    conditioned=trajectory_maps(truth[:,cfg['waypoints']],H,W,distance)
    features=net.pred_features(torch.cat([semantic.expand(len(trajectory),-1,-1,-1),observed],1))
    goal_logits=net.pred_goal(features)
    goal_loss=F.binary_cross_entropy_with_logits(goal_logits,target)*cfg['loss_scale']
    pyramid=[conditioned]+[F.avg_pool2d(conditioned,2**i,2**i) for i in range(1,len(features))]
    path_logits=net.pred_traj([torch.cat([f,w],1) for f,w in zip(features,pyramid)])
    path_loss=F.binary_cross_entropy_with_logits(path_logits,target)*cfg['loss_scale']
    result=dict(loss=goal_loss+path_loss,goal_loss=goal_loss,path_loss=path_loss)
    if return_logits:result.update(goal_logits=goal_logits,path_logits=path_logits,target=target)
    return result

def gradients(net):
    grads=[p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
    if not grads:raise RuntimeError('No predictor gradients')
    if not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Nonfinite gradients')
    missing=[n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]
    if missing:raise RuntimeError('Missing predictor gradients: '+repr(missing))
    norm=torch.stack([g.detach().norm().double()**2 for g in grads]).sum().sqrt()
    return float(norm),sum(int(torch.count_nonzero(g)) for g in grads),sum(g.numel() for g in grads)

def ego_inference_no_GT(net,image,history,cfg,templates):
    """Engineering K1 learned goal -> path, no future argument/teacher forcing.

    Not the full TTST K20 evaluator, not a deployment score or distribution mean.
    """
    assert history.shape[1:]==(8,2)
    net.eval();distance,_=templates
    with torch.no_grad():
        semantic=fixed_semantics(net,image);H,W=semantic.shape[-2:]
        observed=trajectory_maps(history,H,W,distance)
        features=net.pred_features(torch.cat([semantic.expand(len(history),-1,-1,-1),observed],1))
        goals=net.softargmax(net.pred_goal(features)[:,cfg['waypoints']])
        conditions=trajectory_maps(goals,H,W,distance)
        pyramid=[conditions]+[F.avg_pool2d(conditions,2**i,2**i) for i in range(1,len(features))]
        return net.softargmax(net.pred_traj([torch.cat([f,c],1) for f,c in zip(features,pyramid)]))
