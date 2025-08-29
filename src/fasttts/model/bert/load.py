from safetensors.torch import load_file
import torch
from .bert import BertModel
from transformers import AutoConfig
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

def apply_weight(model : BertModel, load_params):
    model_params = list(model.named_parameters()) 
    with torch.no_grad():
        for (name, param) in model_params:
            if "LayerNorm" in name:
                if "weight" in name:
                    name = name.replace("weight", "gamma")
                else:
                    assert "bias" in name
                    name = name.replace("bias", "beta")
            modified_name = "bert.encoder." + name
            if modified_name not in load_params:
                modified_name = "bert." + name
            if modified_name not in load_params:
                assert False
            else:
                param.copy_(load_params[modified_name])
                del load_params[modified_name]

def from_pretrained():
    model = "google-bert/bert-base-uncased"
    config = AutoConfig.from_pretrained(model)
    load_params = load_file(hf_hub_download(repo_id="google-bert/bert-base-uncased", filename="model.safetensors"))
    model = BertModel(config)
    apply_weight(model, load_params)
    model.train(False)
    return model
