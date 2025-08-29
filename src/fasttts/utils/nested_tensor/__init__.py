import torch
from torch.nested._internal.nested_tensor import NestedTensor

def makeLift(f):
    def lift(nt, *args, **kwargs):
        return torch.nested.nested_tensor(f(nt.values(), *args, **kwargs), nt.offsets)
    return lift
def nested_view(t : NestedTensor, *shape):
    return torch.nested.nested_tensor_from_jagged(t.values().view(-1, *shape), t.offsets())

nested_zeros_like = makeLift(torch.zeros_like)