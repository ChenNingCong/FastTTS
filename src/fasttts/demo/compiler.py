from fasttts.model.kokoro.istftnet import Generator
import torch.nn.functional as F
import torch
from fasttts.model.kokoro.pipeline import KPipeline
import torch.nn as nn
import os

# This is copied from Kokoro-TTS's decoder
def forward_inner(self, x, s, har):
    for i in range(self.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1) 
        x_source = self.noise_convs[i](har)
        x_source = self.noise_res[i](x_source, s)
        x = self.ups[i](x)
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)
        x = x + x_source
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[i*self.num_kernels+j](x, s)
            else:
                xs += self.resblocks[i*self.num_kernels+j](x, s)
        x = xs / self.num_kernels
    x = F.leaky_relu(x)
    x = self.conv_post(x)
    return x 

# A wrapper to compile only the forward_inner function
class DynamicModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        s,
        har
    ):
        result = forward_inner(self.model, x, s, har)
        return result

# template arguments
def compile_generator(g : Generator, dtype, device : str,  output_dir : str):
    from torch.export import Dim
    C0 =128
    B = BS
    C= 512
    L0 = 581
    x = torch.randn(B,C,2*L0).cuda().to(dtype)
    s = torch.randn(B,C0).cuda().to(dtype)
    har = torch.randn(B,22, 69721).cuda().to(dtype)
    model = DynamicModel(g).cuda().to(dtype)
    dim_0 = Dim("L0")
    dim_1 = Dim('L0') * 60 + 1
    dynamic_shapes = ((B,C, dim_0), (B,C0,), (B, 22, dim_1))
    with torch.no_grad():
        ep = torch.export.export(model, (x,s,har), dynamic_shapes=dynamic_shapes)
        print(ep)
    import os
    with torch.inference_mode():
        inductor_configs = {}
        if device == "cuda":
            assert torch.cuda.is_available()
            inductor_configs["max_autotune"] = True
        else:
            assert device == "cpu"
        path = torch._inductor.aoti_compile_and_package(
            ep,
            package_path=os.path.join(output_dir, "generator.pt2"),
            inductor_configs=inductor_configs
        )
    return path

# We only quantize the decoder, since it's the most compute-intensive part of the model
class LowprecisionWrapper(nn.Module):
    def __init__(self, model, dtype) -> None:
        super().__init__()
        self.model = model
        self.model = self.model.to(dtype)
        self.cast_dtype = dtype
    def forward(self, *args, **kwargs) -> None:
        args = [i.to(self.cast_dtype) for i in args]
        kwargs = {i:kwargs[i].to(self.cast_dtype) for i in kwargs}
        return self.model.forward(*args, **kwargs).to(torch.float32)

def apply_generator_fix(pipeline:KPipeline, path):
    compiled_model = torch._inductor.aoti_load_package(path)
    pipeline.model.decoder.generator.model.forward_inner = compiled_model

def init(compiled_dir : str, device : str, batched_input : bool = False):
    torch.set_float32_matmul_precision('high')
    from fasttts.model.kokoro.fix import apply_fix
    pipeline = KPipeline(lang_code='a')
    model = pipeline.model
    assert model is not None
    apply_fix(model)
    model.cuda()
    model.decoder.generator = LowprecisionWrapper(pipeline.model.decoder.generator, torch.bfloat16)
    compile_binary = os.path.join(compiled_dir, "generator.pt2")
    if not os.path.exists(compiled_dir):
        raise FileNotFoundError(f"{compiled_dir} is not a valid directory!")
    if not batched_input:
        # currently we don't support compiler for batched input
        if not os.path.exists(compile_binary):
            compile_generator(pipeline.model.decoder.generator.model, dtype=torch.bfloat16, output_dir=compiled_dir, device=device)
        apply_generator_fix(pipeline, compile_binary)
    return pipeline



