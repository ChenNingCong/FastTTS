from .model import KModel
import torch
def af(m):
    try:
        torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')
    except:
        pass
def apply_fix(model : KModel):
    print("Apply fix to remove weight normalization")
    model.apply(af)