from .model import KModel
import torch
def af(m):
    try:
        torch.nn.utils.parametrize.remove_parametrizations(m, 'weight')
    except:
        pass

# from functools import reduce
# def rgetattr(obj, attr, *args):
#     for i in attr.split('.'):
#         obj = getattr(obj, i)
#     return obj
    
# def rsetattr(obj, attr, val):
#     a = attr.split(".")
#     for i in a[:-1]:
#         obj = getattr(obj, i)
#     setattr(obj, a[-1], val)

# def filter_apply(model : torch.nn.Module, f):
#     for child in dict(model.named_modules()).items():
#         if child[0]:
#             try:
#                 module = rgetattr(model, child[0])
#                 maybe_replace = f(module)
#                 if maybe_replace is not None:
#                     print(child[0])
#                     rsetattr(model, child[0], f(module))
#             except Exception as e:
#                 print(e)
    
# def modify_istftnet_ada1n(model : torch.nn.Module):
#     from .istftnet import AdaIN1d, AdaIN1d_fix
#     def replace_fn(m):
#         if isinstance(m, AdaIN1d):
#             print("Replace one ada model")
#             return AdaIN1d_fix(m.fc.in_features, m.fc.out_features)
#     filter_apply(model, replace_fn)

def modify_istftnet_ada1n(model : torch.nn.Module):
    from .istftnet import AdaIN1d
    if isinstance(model, AdaIN1d):
        print("Replace one ada model")
        with torch.no_grad():
            fc = model.fc
            fc1 = model.fc1
            fc2 = model.fc2
            # the matrix is transposed, if we want to split output dim
            # then we need to split the row of underlying weight matrix
            assert fc.weight.shape[0] % 2 == 0
            num_features = fc.weight.shape[0] // 2
            # Split the weight tensor and copy to the new layers
            fc1.weight.copy_(fc.weight[:num_features, :])
            fc2.weight.copy_(fc.weight[num_features:, :])

            # Split the bias tensor (if it exists) and copy
            if fc.bias is not None:
                fc1.bias.copy_(fc.bias[:num_features])
                fc2.bias.copy_(fc.bias[num_features:])
            del model.fc
def apply_fix(model : KModel):
    print("Apply fix to remove weight normalization")
    model.apply(af)
    print("Apply fix to istftnet AdaIN1d")
    model.apply(modify_istftnet_ada1n)