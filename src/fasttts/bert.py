import torch
import torch.nn as nn
from transformers.models.bert import BertConfig
from typing import Optional, List
from .utils.nested_tensor import NestedTensor, nested_zeros_like
from .utils.nested_tensor.jagged_attention import self_attention

class PositionId(nn.Module):
    def __init__(self, parameter : torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("buffer", parameter, persistent=False)
    def forward(self, input_ids : torch.Tensor):
        
        input_lengths = torch.diff(input_ids.offsets()) # type: ignore
        # todo : reduce unnecessary build here
        tmp = torch.nested.nested_tensor([self.buffer[:i] for i in input_lengths]) # type: ignore
        return torch.nested.nested_tensor_from_jagged(tmp.values(), input_ids.offsets()) # type: ignore

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config : BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = PositionId(torch.arange(config.max_position_embeddings))
        self.token_type_ids = PositionId(torch.zeros(config.max_position_embeddings, dtype=torch.long))

    def forward(
        self,
        input_ids: Optional[NestedTensor] = None,
        token_type_ids: Optional[NestedTensor] = None
    ) -> torch.Tensor:
        position_ids = self.position_ids(input_ids)

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                token_type_ids = self.token_type_ids(input_ids)
            else:
                token_type_ids = nested_zeros_like(input_ids, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
class BertSelfOutput(nn.Module):
    def __init__(self, config : BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states




class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        
        assert not config.is_decoder

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        query_layer = mixed_query_layer
        context_layer = self_attention(query_layer, key_layer, value_layer, self.num_attention_heads, self.num_attention_heads, self.attention_head_size)
        return context_layer

from transformers.activations import ACT2FN
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor]:
        # post norm
        return self.output(self.self(hidden_states), hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        assert not self.is_decoder
        self.add_cross_attention = config.add_cross_attention 
        assert not self.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        self_attention_outputs = self.attention(hidden_states)
        layer_output = self.feed_forward_chunk(self_attention_outputs)
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
class BertEncoder(nn.Module):
    def __init__(self, config : BertConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor
    ):
        for _, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
        return hidden_states

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = torch.stack([hidden_states[i, 0] for i in range(hidden_states.shape[0])])
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
class BertModel(nn.Module):
    def __init__(self, config, add_pooling_layer=True):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        """
        super().__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ):
        assert not self.training
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        encoder_outputs = self.encoder(embedding_output)
        pooled_output = self.pooler(encoder_outputs) if self.pooler is not None else None
        if self.pooler is None:
            return encoder_outputs
        else:
            return pooled_output
