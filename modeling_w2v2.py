import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Tuple, Union, Optional, Dict
import warnings

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Attention, 
    Wav2Vec2FeedForward,
    Wav2Vec2EncoderLayer, 
    Wav2Vec2EncoderLayerStableLayerNorm, 
    Wav2Vec2Encoder, 
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2FeatureEncoder,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForXVector,
    AMSoftmaxLoss,
    TDNNLayer,
)

from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    cached_file,
    check_torch_load_is_safe,
    is_peft_available,
    is_safetensors_available,
    is_torch_flex_attn_available,
    logging,
)


WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"

if is_safetensors_available():
    from safetensors.torch import load_file as safe_load_file


if is_torch_flex_attn_available():
    from transformers.integrations.flex_attention import make_flex_block_causal_mask


logger = logging.get_logger(__name__)


_HIDDEN_STATES_START_POSITION = 2

class Wav2Vec2Config(Wav2Vec2Config):
    def __init__(self, 
                 vocab_size=32, 
                 hidden_size=768, 
                 num_hidden_layers=12, 
                 num_attention_heads=12, 
                 intermediate_size=3072, 
                 hidden_act="gelu", 
                 hidden_dropout=0.1, 
                 activation_dropout=0.1, 
                 attention_dropout=0.1, 
                 feat_proj_dropout=0, 
                 feat_quantizer_dropout=0, 
                 final_dropout=0.1, 
                 layerdrop=0.1, 
                 initializer_range=0.02, 
                 layer_norm_eps=0.00001, 
                 feat_extract_norm="group", 
                 feat_extract_activation="gelu", 
                 conv_dim=..., 
                 conv_stride=..., 
                 conv_kernel=..., 
                 conv_bias=False, 
                 num_conv_pos_embeddings=128, 
                 num_conv_pos_embedding_groups=16, 
                 do_stable_layer_norm=False, 
                 apply_spec_augment=True, 
                 mask_time_prob=0.05, 
                 mask_time_length=10, 
                 mask_time_min_masks=2, 
                 mask_feature_prob=0, 
                 mask_feature_length=10, 
                 mask_feature_min_masks=0, 
                 num_codevectors_per_group=320, 
                 num_codevector_groups=2, 
                 contrastive_logits_temperature=0.1, 
                 num_negatives=100, 
                 codevector_dim=256, 
                 proj_codevector_dim=256, 
                 diversity_loss_weight=0.1, 
                 ctc_loss_reduction="sum", 
                 ctc_zero_infinity=False, 
                 use_weighted_layer_sum=False, 
                 classifier_proj_size=256, 
                 tdnn_dim=..., 
                 tdnn_kernel=..., 
                 tdnn_dilation=..., 
                 xvector_output_dim=512, 
                 pad_token_id=0, 
                 bos_token_id=1, 
                 eos_token_id=2, 
                 add_adapter=False, 
                 adapter_kernel_size=3, 
                 adapter_stride=2, 
                 num_adapter_layers=3,
                 output_hidden_size=None, 
                 adapter_attn_dim=None,
                 adapter_fusion = False,
                 fusion_adapter_names=[], 
                 **kwargs):
        super().__init__(vocab_size, 
                         hidden_size, 
                         num_hidden_layers, 
                         num_attention_heads, 
                         intermediate_size, 
                         hidden_act, 
                         hidden_dropout, 
                         activation_dropout, 
                         attention_dropout, 
                         feat_proj_dropout, 
                         feat_quantizer_dropout, 
                         final_dropout, 
                         layerdrop, 
                         initializer_range, 
                         layer_norm_eps, 
                         feat_extract_norm, 
                         feat_extract_activation, 
                         conv_dim, conv_stride, 
                         conv_kernel, conv_bias, 
                         num_conv_pos_embeddings, 
                         num_conv_pos_embedding_groups, 
                         do_stable_layer_norm, 
                         apply_spec_augment, 
                         mask_time_prob, 
                         mask_time_length, 
                         mask_time_min_masks, 
                         mask_feature_prob, 
                         mask_feature_length, 
                         mask_feature_min_masks, 
                         num_codevectors_per_group,
                         num_codevector_groups, 
                         contrastive_logits_temperature, 
                         num_negatives, 
                         codevector_dim, 
                         proj_codevector_dim, 
                         diversity_loss_weight, 
                         ctc_loss_reduction, 
                         ctc_zero_infinity, 
                         use_weighted_layer_sum, 
                         classifier_proj_size, 
                         tdnn_dim, tdnn_kernel, 
                         tdnn_dilation, 
                         xvector_output_dim, 
                         pad_token_id, 
                         bos_token_id, 
                         eos_token_id, 
                         add_adapter, 
                         adapter_kernel_size, 
                         adapter_stride, 
                         num_adapter_layers, 
                         output_hidden_size, 
                         adapter_attn_dim,
                         adapter_fusion,
                         fusion_adapter_names,
                         **kwargs)

@dataclass
@auto_docstring(
    custom_intro="""
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.
    """
)
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    r"""
    loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
        paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
    projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
        projected quantized states.
    projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
        Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
        target vectors for contrastive loss.
    codevector_perplexity (`torch.FloatTensor` of shape `(1,)`):
        The perplexity of the codevector distribution, used to measure the diversity of the codebook.
    contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
        The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """

    loss: Optional[torch.FloatTensor] = None
    projected_states: Optional[torch.FloatTensor] = None
    projected_quantized_states: Optional[torch.FloatTensor] = None
    codevector_perplexity: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None


def _compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://huggingface.co/papers/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.detach().sum(-1).tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def _sample_negative_indices(
    features_shape: tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states

class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class Wav2Vec2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(GradientCheckpointingLayer):
    def __init__(self, config):
        super().__init__()
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
            config=config,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.adapter_layer = None
        self.adapter_fusion_layer = None
        # --- begin new block ---
        if getattr(config, "adapter_attn_dim", None) is not None:
            if getattr(config, "adapter_fusion", False) and getattr(config, "fusion_adapter_names", []):
                self.adapter_fusion_layer = Wav2Vec2AdapterFusionLayer(config, getattr(config, "fusion_adapter_names"))
            else:
                self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        # --- end new block ---

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        #if self.adapter_layer is not None:
        #    hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # --- begin new block ---
        if self.adapter_fusion_layer is not None:
            hidden_states = hidden_states + self.adapter_fusion_layer(hidden_states)
        elif self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)
        # --- end new block ---

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://huggingface.co/papers/1611.01144) for more information.
    """

    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # storage for codebook variables (codewords)
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # can be decayed for training
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            marginal_probs = probs.mean(dim=0)

        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # compute perplexity
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # take argmax in non-differentiable way
            # comptute hard codevector distribution (one hot)
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # use probs to retrieve codevectors
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity

class Wav2Vec2Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        self.layers = nn.ModuleList(Wav2Vec2AdapterLayer(config) for _ in range(config.num_adapter_layers))
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class Wav2Vec2AttnAdapterLayer(nn.Module):
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        super().__init__()
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        self.norm = nn.LayerNorm(self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        self.act_fn = nn.ReLU()
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.norm(hidden_states)

        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states

class Wav2Vec2AdapterFusionLayer(nn.Module):
    """Holds multiple attention adapters and learns softmax fusion weights."""
    def __init__(self, config, adapter_names):
        super().__init__()
        if getattr(config, "adapter_attn_dim", None) is None:
            raise ValueError("adapter_fusion requires config.adapter_attn_dim to be set.")
        if not adapter_names:
            raise ValueError("adapter_fusion enabled but fusion_adapter_names is empty.")

        self.adapter_names = list(adapter_names)
        self.adapters = nn.ModuleDict({
            name: Wav2Vec2AttnAdapterLayer(config) for name in self.adapter_names
        })
        self.fusion_logits = nn.Parameter(torch.zeros(len(self.adapter_names)))

    def forward(self, hidden_states: torch.FloatTensor):
        outs = [self.adapters[name](hidden_states) for name in self.adapter_names]
        stacked = torch.stack(outs, dim=0)
        w = torch.softmax(self.fusion_logits, dim=0).view(-1, 1, 1, 1)
        return (stacked * w).sum(dim=0)

@auto_docstring
class Wav2Vec2PreTrainedModel(PreTrainedModel):
    config: Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # Wav2Vec2ForPreTraining last 2 linear layers need standard Linear init.
        if isinstance(module, Wav2Vec2ForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        # gumbel softmax requires special init
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def _get_adapters(self):
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.")

        adapter_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights[".".join([name, param_name])] = param

        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_weights[".".join(["lm_head", name])] = param

        return adapter_weights

    def init_adapter_layers(self):
        """
        (Re-)initialize attention adapter layers and lm head for adapter-only fine-tuning
        """
        # init attention adapters
        for module in self.modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)

        # init lm head
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)

    def load_adapter(self, target_lang: str, force_load=True, **kwargs):
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"Cannot load_adapter for {target_lang} if `config.adapter_attn_dim` is not defined.")

        if target_lang == self.target_lang and not force_load:
            logger.warning(f"Adapter weights are already set to {target_lang}.")
            return

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        model_path_or_id = self.config._name_or_path
        state_dict = None

        # 1. Let's first try loading a safetensors adapter weight
        if use_safetensors is not False:
            filepath = WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                state_dict = safe_load_file(weight_path)

            except OSError:
                if use_safetensors:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise

            except Exception:
                # For any other exception, we throw a generic error.
                if use_safetensors:
                    raise OSError(
                        f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                        f" directory containing a file named {filepath}."
                    )

        # 2. If this didn't work let's try loading a PyTorch adapter weight
        if state_dict is None:
            filepath = WAV2VEC2_ADAPTER_PT_FILE.format(target_lang)

            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=filepath,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )

                check_torch_load_is_safe()
                state_dict = torch.load(
                    weight_path,
                    map_location="cpu",
                    weights_only=True,
                )

            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise

            except ValueError:
                raise

            except Exception:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{model_path_or_id}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{model_path_or_id}' is the correct path to a"
                    f" directory containing a file named {filepath}."
                )

        adapter_weights = self._get_adapters()
        unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())

        if len(unexpected_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.")
        elif len(missing_keys) > 0:
            raise ValueError(f"The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.")

        # make sure now vocab size is correct
        target_vocab_size = state_dict["lm_head.weight"].shape[0]
        if target_vocab_size != self.config.vocab_size:
            self.lm_head = nn.Linear(
                self.config.output_hidden_size, target_vocab_size, device=self.device, dtype=self.dtype
            )
            self.config.vocab_size = target_vocab_size

        # make sure that adapter weights are put in exactly the same precision and device placement and overwritten adapter weights
        state_dict = {k: v.to(adapter_weights[k]) for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

        # set target language correctly
        self.target_lang = target_lang

    # --- Adapter Fusion helpers ---
    def _get_adapter_fusions(self):
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} has no adapter fusion layers. Make sure to define `config.adapter_attn_dim`.")

        adapter_fusion_weights = {}
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AdapterFusionLayer):
                for param_name, param in module.named_parameters():
                    adapter_fusion_weights[".".join([name, param_name])] = param

        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_fusion_weights[".".join(["lm_head", name])] = param

        return adapter_fusion_weights

    def init_adapter_fusion_layers(self):
        """
        (Re-)initialize attention adapter layers and lm head for adapter-fusion-only fine-tuning
        """
        # init attention adapter fusion
        for module in self.modules():
            if isinstance(module, Wav2Vec2AdapterFusionLayer):
                self._init_weights(module)

        # init lm head
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)

    def enable_adapter_fusion(self, adapter_names: List[str]):
        """Turn on adapter fusion across all encoder layers.

        This updates self.config flags and (re)creates fusion layers where
        applicable. Call before training/eval. If you call this after a model
        has been created, it will attach fusion modules lazily where missing.
        """
        if not isinstance(adapter_names, (list, tuple)) or len(adapter_names) == 0:
            raise ValueError("adapter_names must be a non-empty list of strings")
        self.config.adapter_fusion = True
        self.config.fusion_adapter_names = list(adapter_names)

        # Walk encoder layers and ensure fusion layer exists
        encoder = getattr(getattr(self, "wav2vec2", self), "encoder", None)
        if encoder is None:
            return
        for layer in encoder.layers:
            if hasattr(layer, "adapter_fusion_layer"):
                if layer.adapter_fusion_layer is None:
                    layer.adapter_fusion_layer = Wav2Vec2AdapterFusionLayer(self.config, self.config.fusion_adapter_names)
                else:
                    # ensure names are in sync
                    if layer.adapter_fusion_layer.adapter_names != self.config.fusion_adapter_names:
                        layer.adapter_fusion_layer = Wav2Vec2AdapterFusionLayer(self.config, self.config.fusion_adapter_names)
            else:
                # Older variant: add attribute dynamically
                layer.adapter_fusion_layer = Wav2Vec2AdapterFusionLayer(self.config, self.config.fusion_adapter_names)

    def _remap_adapter_keys_for_fusion(self, state_dict: Dict[str, torch.Tensor], name: str) -> Dict[str, torch.Tensor]:
        """Remap single-adapter keys to fusion adapter ModuleDict slot.

        Example:
          wav2vec2.encoder.layers.0.adapter_layer.linear_1.weight ->
          wav2vec2.encoder.layers.0.adapter_fusion_layer.adapters.{name}.linear_1.weight
        """
        remapped = {}
        for k, v in state_dict.items():
            if ".adapter_layer." in k and ".encoder.layers." in k:
                remapped[k.replace(".adapter_layer.", f".adapter_fusion_layer.adapters.{name}.")] = v
            elif k.startswith("lm_head."):
                # We skip lm_head by default; if the size matches current, you could set it explicitly.
                # remapped[k] = v  # uncomment to allow overwrite when sizes match
                continue
            else:
                # keep unrelated keys (rare), they will be ignored by strict=False
                remapped[k] = v
        return remapped

    def load_adapter_as(self, target_lang: str, name: str, force_load=True, **kwargs):
        """Load a language adapter and register it into the fusion slot under `name`.

        Internally reuses load_adapter's file resolution, then remaps keys into
        adapter_fusion submodules. Ensure you called enable_adapter_fusion([...])
        beforehand (or pass a name list including `name`).
        """
        if getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError("Cannot load adapter if config.adapter_attn_dim is not defined.")
        if not getattr(self.config, "adapter_fusion", False):
            # Implicitly enable fusion with the single name if not enabled yet
            self.enable_adapter_fusion([name])

        # Reuse upstream loader to get state_dict (without assigning)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        # Borrow logic from load_adapter but stop right after getting `state_dict`.
        model_path_or_id = self.config._name_or_path
        state_dict = None

        if is_safetensors_available() and use_safetensors is not False:
            WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"
            try:
                from safetensors.torch import load_file as safe_load_file
                weight_path = cached_file(
                    model_path_or_id,
                    filename=WAV2VEC2_ADAPTER_SAFE_FILE.format(target_lang),
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                state_dict = safe_load_file(weight_path)
            except Exception:
                state_dict = None

        if state_dict is None:
            WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
            try:
                weight_path = cached_file(
                    model_path_or_id,
                    filename=WAV2VEC2_ADAPTER_PT_FILE.format(target_lang),
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                check_torch_load_is_safe()
                state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            except Exception as e:
                raise e

        # Remap keys into fusion namespace
        remapped = self._remap_adapter_keys_for_fusion(state_dict, name)

        # Ensure fusion modules exist (in case called before forward)
        self.enable_adapter_fusion(list({*getattr(self.config, "fusion_adapter_names", []), name}))

        # Load with strict=False (ignore unrelated keys)
        missing_keys, unexpected_keys = self.load_state_dict(remapped, strict=False)

        # Optional: sanity logging
        if len(unexpected_keys) > 0:
            warnings.warn(f"Unexpected keys while loading fusion adapter '{name}': {sorted(list(unexpected_keys))[:5]} ...")
        elif len(missing_keys) > 0:
            warnings.warn(f"Unexpected keys while loading fusion adapter '{name}': {sorted(list(missing_keys))[:5]} ...")
        return True
    
        #adapter_weights = self._get_adapters()
        #unexpected_keys = set(state_dict.keys()) - set(adapter_weights.keys())
        #missing_keys = set(adapter_weights.keys()) - set(state_dict.keys())

        #if len(unexpected_keys) > 0:
        #    raise ValueError(f"The adapter weights {weight_path} has unexpected keys: {', '.join(unexpected_keys)}.")
        #elif len(missing_keys) > 0:
        #    raise ValueError(f"The adapter weights {weight_path} has missing keys: {', '.join(missing_keys)}.")

@auto_docstring
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.Tensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://huggingface.co/papers/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Wav2Vec2BaseModelOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    Wav2Vec2 Model with a quantizer and `VQ` head on top.
    """
)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # Initialize weights and apply final processing
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, Wav2Vec2ForPreTrainingOutput]:
        r"""
        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        sampled_negative_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_negatives)`, *optional*):
            Indices indicating which quantized target vectors are used as negative sampled vectors in contrastive loss.
            Required input for pre-training.

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
        >>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        >>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  # Batch size 1

        >>> # compute masked indices
        >>> batch_size, raw_sequence_length = input_values.shape
        >>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
        >>> mask_time_indices = _compute_mask_indices(
        ...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
        ... )
        >>> sampled_negative_indices = _sample_negative_indices(
        ...     features_shape=(batch_size, sequence_length),
        ...     num_negatives=model.config.num_negatives,
        ...     mask_time_indices=mask_time_indices,
        ... )
        >>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
        >>> sampled_negative_indices = torch.tensor(
        ...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
        ... )

        >>> with torch.no_grad():
        ...     outputs = model(input_values, mask_time_indices=mask_time_indices)

        >>> # compute cosine similarity between predicted (=projected_states) and target (=projected_quantized_states)
        >>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

        >>> # show that cosine similarity is much higher than random
        >>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
        tensor(True)

        >>> # for contrastive loss training model should be put into train mode
        >>> model = model.train()
        >>> loss = model(
        ...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
        ... ).loss
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, mask_time_indices=mask_time_indices
        )

        quantized_features = quantized_features.to(self.project_q.weight.dtype)
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://huggingface.co/papers/2006.11477
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
            target = ((1 - mask_time_indices.long()) * -100).transpose(0, 1).flatten()

            contrastive_loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )

@auto_docstring
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        warnings.warn(
            "The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.", FutureWarning
        )

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


@auto_docstring(
    custom_intro="""
    Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
    """
)
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        r"""
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`Wav2Vec2ForCTC`] with adapters. Uses 'eng' by
            default.
        """
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # Note that `tie_weights` is usually used to tie input and output embedding weights. The method is re-purposed to
        # correctly load adapter layers for Wav2Vec2 so that we do not have to introduce a new API to
        # [`PreTrainedModel`]. While slightly hacky, Wav2Vec2 never has to tie input and output embeddings, so that it is
        # ok to repurpose this function here.
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

@auto_docstring(
    custom_intro="""
    Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """
)
class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
            (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            expand_padding_mask = padding_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, TokenClassifierOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
            (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

        if is_peft_available():
            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # for backward compatibility, we keep nn.Linear but call F.conv1d for speed up
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.activation(hidden_states)
        return hidden_states


@auto_docstring(
    custom_intro="""
    Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """
)
class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @auto_docstring
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, XVectorOutput]:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library
            (`pip install torchcodec`) or the soundfile library (`pip install soundfile`).
            To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and conversion
            into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
__all__ = [
    "Wav2Vec2ForAudioFrameClassification",
    "Wav2Vec2ForCTC",
    "Wav2Vec2ForMaskedLM",
    "Wav2Vec2ForPreTraining",
    "Wav2Vec2ForSequenceClassification",
    "Wav2Vec2ForXVector",
    "Wav2Vec2Model",
    "Wav2Vec2PreTrainedModel",
]