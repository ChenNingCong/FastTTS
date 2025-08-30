from .istftnet import Decoder
from .modules import CustomAlbert, ProsodyPredictor, TextEncoder
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union, List
import json
import torch

class KModel(torch.nn.Module):
    '''
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    '''

    MODEL_NAMES = {
        'hexgrad/Kokoro-82M': 'kokoro-v1_0.pth',
        'hexgrad/Kokoro-82M-v1.1-zh': 'kokoro-v1_1-zh.pth',
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False
    ):
        super().__init__()
        if repo_id is None:
            repo_id = 'hexgrad/Kokoro-82M'
            print(f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning.")
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename='config.json')
            with open(config, 'r', encoding='utf-8') as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config['vocab']
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config['n_token'], **config['plbert']))
        self.bert_encoder = torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])
        self.context_length = self.bert.config.max_position_embeddings
        self.predictor = ProsodyPredictor(
            style_dim=config['style_dim'], d_hid=config['hidden_dim'],
            nlayers=config['n_layer'], max_dur=config['max_dur'], dropout=config['dropout']
        )
        self.text_encoder = TextEncoder(
            channels=config['hidden_dim'], kernel_size=config['text_encoder_kernel_size'],
            depth=config['n_layer'], n_symbols=config['n_token']
        )
        self.decoder = Decoder(
            dim_in=config['hidden_dim'], style_dim=config['style_dim'],
            dim_out=config['n_mels'], disable_complex=disable_complex, **config['istftnet']
        )
        if not model:
            model = hf_hub_download(repo_id=repo_id, filename=KModel.MODEL_NAMES[repo_id])
        for key, state_dict in torch.load(model, map_location='cpu', weights_only=True).items():
            assert hasattr(self, key), key
            try:
                getattr(self, key).load_state_dict(state_dict)
            except:
                logger.debug(f"Did not load {key} from state_dict")
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                getattr(self, key).load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    def forward_bert(self, input_ids, text_mask):
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2) # (B, C, L)
        return d_en
    
    def forward_duration(self, input_ids, input_lengths, text_mask, ref_s, speed, d_en):
        s = ref_s[:, 128:] # (B, C0)
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask) # (B, L, C + C0)
        # d : [batch_size, max_seq_len, hidden_size]
        x, _ = self.predictor.lstm(d) # (B, L, C)
        duration = self.predictor.duration_proj(x) # (B, L, C2)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed # (B, L)
        pred_dur = torch.round(duration).clamp(min=1).long() # (B, L)
        # indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=self.device), pred_dur)
        indices = [torch.repeat_interleave(torch.arange(input_lengths[i].item(), device=self.device), pred_dur[i][:input_lengths[i].item()]) for i in range(input_ids.shape[0])] # (B, L0)
        max_indice_len = max([len(i) for i in indices])    
        pred_aln_trg = torch.zeros((*input_ids.shape, max_indice_len), device=self.device)  # (B, L, L0)
        for i in range(input_ids.shape[0]):
            pred_aln_trg[i, indices[i], torch.arange(len(indices[i]))] = 1
            # pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        # pred_aln_trg : [batch_size, max_seq_len, max_indice_len]
        en = torch.bmm(d.transpose(-1, -2), pred_aln_trg) # (B, C+C0, L0)
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s) # (B, 2*L0), (B, 2*L0)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask) # (B, C, L)
        asr = t_en @ pred_aln_trg # (B, C, L0)
        return asr, F0_pred, N_pred, pred_dur
    
    def forward_decoder(self, asr, F0_pred, N_pred, ref_s):
        return self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze(0)
    
    @torch.no_grad()
    def forward_with_tokens(
        self,
        input_ids: torch.LongTensor,
        input_lengths: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        max_len = input_lengths.max().item()

        text_mask = torch.arange(max_len).unsqueeze(0).to(self.device)
        text_mask = text_mask.expand(len(input_ids), -1)
        text_mask = torch.gt(text_mask, input_lengths.unsqueeze(1) - 1)

        dn = self.forward_bert(input_ids=input_ids, text_mask=text_mask)
        asr, F0_pred, N_pred, pred_dur = self.forward_duration(input_ids=input_ids, 
                                                     input_lengths=input_lengths, 
                                                     text_mask=text_mask, 
                                                     ref_s=ref_s, 
                                                     speed=speed, d_en=dn)
        audio =  self.forward_decoder(asr=asr, F0_pred=F0_pred, N_pred=N_pred, ref_s=ref_s)
        return audio, pred_dur

    def forward(
        self,
        phonemes: List[str],
        ref_s: List[torch.FloatTensor],
        speed: float = 1,
        return_output: bool = False
    ) -> Union[List['KModel.Output'], List[torch.Tensor]]:
        def fn(phonemes : str) -> torch.LongTensor:
            input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
            logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
            assert len(input_ids)+2 <= self.context_length, (len(input_ids)+2, self.context_length)
            input_ids = torch.tensor([0, *input_ids, 0], device=self.device)
            return input_ids
        input_id_list = [fn(i) for i in phonemes]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_id_list, batch_first=True, padding_value=0)
        input_lengths = torch.tensor([len(i) for i in input_id_list], device=self.device)
        ref_s = torch.cat(ref_s, 0).to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, input_lengths, ref_s, speed)
        audio = audio.cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return [self.Output(audio=audio[i], pred_dur=pred_dur[i]) for i in range(len(phonemes))] if return_output else [audio[i] for i in range(len(phonemes))]

class KModelForONNX(torch.nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        waveform, duration = self.kmodel.forward_with_tokens(input_ids, ref_s, speed)
        return waveform, duration
