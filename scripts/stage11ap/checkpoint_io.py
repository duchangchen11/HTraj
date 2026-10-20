"""Read trusted author legacy modules with an explicit class allowlist.

Try weights_only=True first. Old torch.save(module) has persistent module IDs
unsupported by that loader. Fallback remains restricted by find_class, instead
of unrestricted pickle. Official frozen bytes are never modified.
"""
import pickle, importlib, types
import torch
GLOBALS={
 'segmentation_models_pytorch.unet.model.Unet',
 'segmentation_models_pytorch.encoders.resnet.ResNetEncoder',
 'segmentation_models_pytorch.unet.decoder.UnetDecoder',
 'segmentation_models_pytorch.unet.decoder.DecoderBlock',
 'segmentation_models_pytorch.base.modules.Conv2dReLU',
 'segmentation_models_pytorch.base.modules.Attention',
 'segmentation_models_pytorch.base.modules.Activation',
 'segmentation_models_pytorch.base.heads.SegmentationHead',
 'torchvision.models.resnet.Bottleneck',
 'torch.nn.modules.conv.Conv2d','torch.nn.modules.batchnorm.BatchNorm2d',
 'torch.nn.modules.activation.ReLU','torch.nn.modules.activation.Softmax',
 'torch.nn.modules.pooling.MaxPool2d','torch.nn.modules.container.Sequential',
 'torch.nn.modules.container.ModuleList','torch.nn.modules.linear.Identity',
 'torch._utils._rebuild_parameter','torch._utils._rebuild_tensor_v2',
 'torch.FloatStorage','torch.LongStorage','collections.OrderedDict','builtins.set',
}
class RestrictedCheckpointUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        key=module+'.'+name
        if key not in GLOBALS:raise pickle.UnpicklingError('Checkpoint global blocked: '+key)
        return getattr(importlib.import_module(module),name)

def restricted_load(file,**kwargs):return RestrictedCheckpointUnpickler(file,**kwargs).load()
restricted_pickle=types.ModuleType('stage11ap_restricted_pickle')
restricted_pickle.Unpickler=RestrictedCheckpointUnpickler
restricted_pickle.load=restricted_load

def load_author_segmentation(path):
    # Known trusted official Drive bytes only: the caller verifies SHA against
    # the already CRC-tested archive before allowing this operation.
    classes=[getattr(importlib.import_module(s.rsplit('.',1)[0]),s.rsplit('.',1)[1]) for s in sorted(GLOBALS)]
    try:
        with torch.serialization.safe_globals(classes):
            return torch.load(path,map_location='cpu',weights_only=True),None
    except pickle.UnpicklingError as e:
        first=str(e)
    obj=torch.load(path,map_location='cpu',weights_only=False,pickle_module=restricted_pickle)
    return obj,first
