#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ContextManager, Iterator, cast, Optional, Dict, Sequence

import numpy as np
import torch
from gguf import GGUFWriter, WriterState, Keys, GGUF_VERSION, GGUFValueType
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from torch import Tensor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


###### MODEL DEFINITIONS ######

class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class GGUFParamStore:
    def __init__(self, arch: str):
        # not calling super.__init__() on purpose
        self.arch = arch
        self.kv_data: Dict[str, Any] = {}
        self.kv_types: Dict[str, GGUFValueType] = {}
        self.add_string(Keys.General.ARCHITECTURE, self.arch)

    def get_params_dict(self) -> Dict[str, Any]:
        retval = {"gguf_version": GGUF_VERSION,
                  **self.kv_data}
        return retval

    def add_uint8(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.UINT8

    def add_int8(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.INT8

    def add_uint16(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.UINT16

    def add_int16(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.INT16

    def add_uint32(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.UINT32

    def add_int32(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.INT32

    def add_float32(self, key: str, val: float) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.FLOAT32

    def add_uint64(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.UINT64

    def add_int64(self, key: str, val: int) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.INT64

    def add_float64(self, key: str, val: float) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.FLOAT64

    def add_bool(self, key: str, val: bool) -> None:
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.BOOL

    def add_string(self, key: str, val: str) -> None:
        if not val:
            return
        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.STRING

    def add_array(self, key: str, val: Sequence[Any]) -> None:
        if not isinstance(val, Sequence):
            raise ValueError("Value must be a sequence for array type")

        self.kv_data[key] = val
        self.kv_types[key] = GGUFValueType.ARRAY

    # Specialized add_* functions start here
    def add_architecture(self) -> None:
        self.add_string(Keys.General.ARCHITECTURE, self.arch)

    def add_author(self, author: str) -> None:
        self.add_string(Keys.General.AUTHOR, author)

    def add_tensor_data_layout(self, layout: str) -> None:
        self.add_string(Keys.LLM.TENSOR_DATA_LAYOUT.format(arch=self.arch), layout)

    def add_url(self, url: str) -> None:
        self.add_string(Keys.General.URL, url)

    def add_description(self, description: str) -> None:
        self.add_string(Keys.General.DESCRIPTION, description)

    def add_source_url(self, url: str) -> None:
        self.add_string(Keys.General.SOURCE_URL, url)

    def add_source_hf_repo(self, repo: str) -> None:
        self.add_string(Keys.General.SOURCE_HF_REPO, repo)

    def add_file_type(self, ftype: int) -> None:
        self.add_uint32(Keys.General.FILE_TYPE, ftype)

    def add_name(self, name: str) -> None:
        self.add_string(Keys.General.NAME, name)

    def add_quantization_version(self, quantization_version: GGMLQuantizationType) -> None:
        self.add_uint32(
            Keys.General.QUANTIZATION_VERSION, quantization_version)

    def add_custom_alignment(self, alignment: int) -> None:
        self.data_alignment = alignment
        self.add_uint32(Keys.General.ALIGNMENT, alignment)

    def add_context_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.CONTEXT_LENGTH.format(arch=self.arch), length)

    def add_embedding_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.EMBEDDING_LENGTH.format(arch=self.arch), length)

    def add_block_count(self, length: int) -> None:
        self.add_uint32(Keys.LLM.BLOCK_COUNT.format(arch=self.arch), length)

    def add_feed_forward_length(self, length: int) -> None:
        self.add_uint32(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=self.arch), length)

    def add_parallel_residual(self, use: bool) -> None:
        self.add_bool(Keys.LLM.USE_PARALLEL_RESIDUAL.format(arch=self.arch), use)

    def add_head_count(self, count: int) -> None:
        self.add_uint32(Keys.Attention.HEAD_COUNT.format(arch=self.arch), count)

    def add_head_count_kv(self, count: int) -> None:
        self.add_uint32(Keys.Attention.HEAD_COUNT_KV.format(arch=self.arch), count)

    def add_max_alibi_bias(self, bias: float) -> None:
        self.add_float32(Keys.Attention.MAX_ALIBI_BIAS.format(arch=self.arch), bias)

    def add_clamp_kqv(self, value: float) -> None:
        self.add_float32(Keys.Attention.CLAMP_KQV.format(arch=self.arch), value)

    def add_layer_norm_eps(self, value: float) -> None:
        self.add_float32(Keys.Attention.LAYERNORM_EPS.format(arch=self.arch), value)

    def add_layer_norm_rms_eps(self, value: float) -> None:
        self.add_float32(Keys.Attention.LAYERNORM_RMS_EPS.format(arch=self.arch), value)

    def add_rope_dimension_count(self, count: int) -> None:
        self.add_uint32(Keys.Rope.DIMENSION_COUNT.format(arch=self.arch), count)

    def add_rope_freq_base(self, value: float) -> None:
        self.add_float32(Keys.Rope.FREQ_BASE.format(arch=self.arch), value)

    def add_rope_scaling_type(self, value: RopeScalingType) -> None:
        self.add_string(Keys.Rope.SCALING_TYPE.format(arch=self.arch), value.value)

    def add_rope_scaling_factor(self, value: float) -> None:
        self.add_float32(Keys.Rope.SCALING_FACTOR.format(arch=self.arch), value)

    def add_rope_scaling_orig_ctx_len(self, value: int) -> None:
        self.add_uint32(Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=self.arch), value)

    def add_rope_scaling_finetuned(self, value: bool) -> None:
        self.add_bool(Keys.Rope.SCALING_FINETUNED.format(arch=self.arch), value)

    def add_tokenizer_model(self, model: str) -> None:
        self.add_string(Keys.Tokenizer.MODEL, model)

    def add_token_list(self, tokens: Sequence[str] | Sequence[bytes] | Sequence[bytearray]) -> None:
        self.add_array(Keys.Tokenizer.LIST, tokens)

    def add_token_merges(self, merges: Sequence[str] | Sequence[bytes] | Sequence[bytearray]) -> None:
        self.add_array(Keys.Tokenizer.MERGES, merges)

    def add_token_types(self, types: Sequence[TokenType] | Sequence[int]) -> None:
        self.add_array(Keys.Tokenizer.TOKEN_TYPE, types)

    def add_token_scores(self, scores: Sequence[float]) -> None:
        self.add_array(Keys.Tokenizer.SCORES, scores)

    def add_bos_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.BOS_ID, id)

    def add_eos_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.EOS_ID, id)

    def add_unk_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.UNK_ID, id)

    def add_sep_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.SEP_ID, id)

    def add_pad_token_id(self, id: int) -> None:
        self.add_uint32(Keys.Tokenizer.PAD_ID, id)

    def add_add_bos_token(self, value: bool) -> None:
        self.add_bool(Keys.Tokenizer.ADD_BOS, value)

    def add_add_eos_token(self, value: bool) -> None:
        self.add_bool(Keys.Tokenizer.ADD_EOS, value)

    def add_chat_template(self, value: str) -> None:
        self.add_string(Keys.Tokenizer.CHAT_TEMPLATE, value)



class GGUFExportedModelDescriptor:
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.ftype = gguf.GGMLQuantizationType.F32
        self.hparams = self.model.config.to_dict()
        self.model_arch = self._get_model_architecture()
        self.gguf_param_store = GGUFParamStore(gguf.MODEL_ARCH_NAMES[self.model_arch])

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        for name, data in self.model.state_dict().items():
            yield name, data

    def set_gguf_parameters(self):
        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_block_count(self.hparams.get(
            "n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")),
        ))
        if (n_ctx := self.hparams.get("max_position_embeddings")) is not None:
            self.gguf_param_store.add_context_length(n_ctx)
        if (n_embd := self.hparams.get("hidden_size")) is not None:
            self.gguf_param_store.add_embedding_length(n_embd)
        if (n_ff := self.hparams.get("intermediate_size")) is not None:
            self.gguf_param_store.add_feed_forward_length(n_ff)
        if (n_head := self.hparams.get("num_attention_heads")) is not None:
            self.gguf_param_store.add_head_count(n_head)
        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_param_store.add_head_count_kv(n_head_kv)

        if (n_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_param_store.add_layer_norm_rms_eps(n_rms_eps)
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_param_store.add_expert_count(n_experts)
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_param_store.add_expert_used_count(n_experts_used)

        self.gguf_param_store.add_parallel_residual(self.hparams.get("use_parallel_residual", True))

    def get_tensor_name_map(self) -> Dict[str, str]:
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        retval = {}
        for torch_name, data_torch in self.get_tensors():
            # we don't need these
            if torch_name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            # map tensor names
            gguf_name = tensor_map.get_name(torch_name, try_suffixes=(".weight", ".bias"))
            if gguf_name is None:
                print(f"Can not map tensor {torch_name!r}")
                import pdb
                pdb.set_trace()
            retval[torch_name] = gguf_name
        return retval

    def get_params_dict(self) -> Dict[str, Any]:
        retval = {"tensor_name_map": self.get_tensor_name_map()}
        retval.update(self.gguf_param_store.get_params_dict())
        return retval

    @staticmethod
    def from_model_architecture(model_architecture):
        if model_architecture == "GPTNeoXForCausalLM":
            return GPTNeoXModel
        if model_architecture == "BloomForCausalLM":
            return BloomModel
        if model_architecture == "MPTForCausalLM":
            return MPTModel
        if model_architecture in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return BaichuanModel
        if model_architecture in ("FalconForCausalLM", "RWForCausalLM"):
            return FalconModel
        if model_architecture == "GPTBigCodeForCausalLM":
            return StarCoderModel
        if model_architecture == "GPTRefactForCausalLM":
            return RefactModel
        if model_architecture == "PersimmonForCausalLM":
            return PersimmonModel
        if model_architecture in ("StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM"):
            return StableLMModel
        if model_architecture == "QWenLMHeadModel":
            return QwenModel
        if model_architecture == "MixtralForCausalLM":
            return MixtralModel
        if model_architecture == "GPT2LMHeadModel":
            return GPT2Model
        if model_architecture == "PhiForCausalLM":
            return Phi2Model
        if model_architecture == "PlamoForCausalLM":
            return PlamoModel
        return GGUFExportedModelDescriptor

    def _get_model_architecture(self) -> gguf.MODEL_ARCH:
        arch = self.hparams["architectures"][0]
        if arch == "GPTNeoXForCausalLM":
            return gguf.MODEL_ARCH.GPTNEOX
        if arch == "BloomForCausalLM":
            return gguf.MODEL_ARCH.BLOOM
        if arch == "MPTForCausalLM":
            return gguf.MODEL_ARCH.MPT
        if arch in ("BaichuanForCausalLM", "BaiChuanForCausalLM"):
            return gguf.MODEL_ARCH.BAICHUAN
        if arch in ("FalconForCausalLM", "RWForCausalLM"):
            return gguf.MODEL_ARCH.FALCON
        if arch == "GPTBigCodeForCausalLM":
            return gguf.MODEL_ARCH.STARCODER
        if arch == "GPTRefactForCausalLM":
            return gguf.MODEL_ARCH.REFACT
        if arch == "PersimmonForCausalLM":
            return gguf.MODEL_ARCH.PERSIMMON
        if arch in ("StableLMEpochForCausalLM", "LlavaStableLMEpochForCausalLM"):
            return gguf.MODEL_ARCH.STABLELM
        if arch == "QWenLMHeadModel":
            return gguf.MODEL_ARCH.QWEN
        if arch == "MixtralForCausalLM":
            return gguf.MODEL_ARCH.LLAMA
        if arch == "GPT2LMHeadModel":
            return gguf.MODEL_ARCH.GPT2
        if arch == "PhiForCausalLM":
            return gguf.MODEL_ARCH.PHI2
        if arch == "PlamoForCausalLM":
            return gguf.MODEL_ARCH.PLAMO

        raise NotImplementedError(f'Architecture "{arch}" not supported!')


class GPTNeoXModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]

        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_param_store.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_param_store.add_rope_dimension_count(
            int(self.hparams["rotary_pct"] * (self.hparams["hidden_size"] // self.hparams["num_attention_heads"])),
        )
        self.gguf_param_store.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_param_store.add_parallel_residual(self.hparams.get("use_parallel_residual", True))
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_eps"])


class BloomModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        self.gguf_param_store.add_name("Bloom")
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        self.gguf_param_store.add_context_length(self.hparams.get("seq_length", n_embed))
        self.gguf_param_store.add_embedding_length(n_embed)
        self.gguf_param_store.add_feed_forward_length(4 * n_embed)
        self.gguf_param_store.add_block_count(self.hparams["n_layer"])
        self.gguf_param_store.add_head_count(n_head)
        self.gguf_param_store.add_head_count_kv(n_head)
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams["n_layer"]
        tensors = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        has_lm_head = True
        n_head = self.hparams.get("n_head", self.hparams.get("num_attention_heads"))
        n_embed = self.hparams.get("hidden_size", self.hparams.get("n_embed"))

        for name, data_torch in tensors.items():
            if "lm_head.weight" not in tensors.keys() and "output.weight" not in tensors.keys():
                has_lm_head = False

            name = re.sub(r'transformer\.', '', name)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            if re.match(r"h\.\d+\.self_attention\.query_key_value\.weight", name):
                # Map bloom-style qkv_linear to gpt-style qkv_linear
                # bloom: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py#L238-L252  # noqa
                # gpt-2: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L312  # noqa
                qkv_weights = data.reshape((n_head, 3, n_embed // n_head, n_embed))
                data = np.concatenate(
                    (
                        qkv_weights[:, 0, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 1, :, :].reshape((-1, n_embed)),
                        qkv_weights[:, 2, :, :].reshape((-1, n_embed)),
                    ),
                    axis=0,
                )
                print("re-format attention.linear_qkv.weight")
            elif re.match(r"h\.\d+\.self_attention\.query_key_value\.bias", name):
                qkv_bias = data.reshape((n_head, 3, n_embed // n_head))
                data = np.concatenate(
                    (
                        qkv_bias[:, 0, :].reshape((n_embed,)),
                        qkv_bias[:, 1, :].reshape((n_embed,)),
                        qkv_bias[:, 2, :].reshape((n_embed,)),
                    ),
                    axis=0,
                )
                print("re-format attention.linear_qkv.bias")

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"=> {new_name}, shape = {data.shape}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)

            if not has_lm_head and name == "word_embeddings.weight":
                self.gguf_param_store.add_tensor("output.weight", data)
                print(name, f"=> output.weight, shape = {data.shape}, {old_dtype} --> {data.dtype}")


class MPTModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layers"]
        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_context_length(self.hparams["max_seq_len"])
        self.gguf_param_store.add_embedding_length(self.hparams["d_model"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_feed_forward_length(4 * self.hparams["d_model"])
        self.gguf_param_store.add_head_count(self.hparams["n_heads"])
        if kv_n_heads := self.hparams["attn_config"].get("kv_n_heads"):
            self.gguf_param_store.add_head_count_kv(kv_n_heads)
        self.gguf_param_store.add_layer_norm_eps(1e-5)
        if self.hparams["attn_config"]["clip_qkv"] is not None:
            self.gguf_param_store.add_clamp_kqv(self.hparams["attn_config"]["clip_qkv"])
        self.gguf_param_store.add_max_alibi_bias(self.hparams["attn_config"]["alibi_bias_max"])

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq")):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            if "scales" in name:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias", ".scales"))
                new_name = new_name.replace("scales", "act.scales")
            else:
                new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)

            # note: MPT output is tied to (same as) wte in original model;
            # for easier implementation in llama.cpp it's duplicated in GGUF, though :/
            if new_name == "token_embd.weight":
                self.gguf_param_store.add_tensor("output.weight", data)


class BaichuanModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)
        hf_repo = self.hparams.get("_name_or_path", "")

        ctx_length = 0
        if "max_sequence_length" in self.hparams:
            ctx_length = self.hparams["max_sequence_length"]
        elif "max_position_embeddings" in self.hparams:
            ctx_length = self.hparams["max_position_embeddings"]
        elif "model_max_length" in self.hparams:
            ctx_length = self.hparams["model_max_length"]
        else:
            print("gguf: can not find ctx length parameter.")
            sys.exit()

        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_source_hf_repo(hf_repo)
        self.gguf_param_store.add_tensor_data_layout("Meta AI original pth")
        self.gguf_param_store.add_context_length(ctx_length)
        self.gguf_param_store.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_param_store.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_param_store.add_head_count(head_count)
        self.gguf_param_store.add_head_count_kv(head_count_kv)
        self.gguf_param_store.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

        if self.hparams.get("rope_scaling") is not None and "factor" in self.hparams["rope_scaling"]:
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_param_store.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_param_store.add_rope_scaling_factor(self.hparams["rope_scaling"]["factor"])

    def write_tensors(self):
        # Collect tensors from generator object
        model_kv = dict(self.get_tensors())
        block_count = self.hparams["num_hidden_layers"]
        head_count = self.hparams["num_attention_heads"]
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        head_count_kv = self.hparams.get("num_key_value_heads", head_count)

        for i in range(block_count):
            if (w := model_kv.get(f"model.layers.{i}.self_attn.W_pack.weight")) is not None:
                print(f"Unpacking and permuting layer {i}")
                model_kv[f"model.layers.{i}.self_attn.q_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 0, head_count, head_count)
                model_kv[f"model.layers.{i}.self_attn.k_proj.weight"] = \
                    self._reverse_hf_permute_part(w, 1, head_count, head_count_kv)
                model_kv[f"model.layers.{i}.self_attn.v_proj.weight"] = \
                    self._reverse_hf_part(w, 2)
                del model_kv[f"model.layers.{i}.self_attn.W_pack.weight"]

        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{name} -> {new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_param_store.add_tensor(new_name, data)

    def _reverse_hf_permute(self, weights: Tensor, n_head: int, n_kv_head: int | None = None) -> Tensor:
        if n_kv_head is not None and n_head != n_kv_head:
            n_head //= n_kv_head

        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    def _reverse_hf_permute_part(
        self, weights: Tensor, n_part: int, n_head: int, n_head_kv: int | None = None,
    ) -> Tensor:
        r = weights.shape[0] // 3
        return self._reverse_hf_permute(weights[r * n_part:r * n_part + r, ...], n_head, n_head_kv)

    def _reverse_hf_part(self, weights: Tensor, n_part: int) -> Tensor:
        r = weights.shape[0] // 3
        return weights[r * n_part:r * n_part + r, ...]


class FalconModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        self.gguf_param_store.add_name("Falcon")
        self.gguf_param_store.add_context_length(2048)  # not in config.json
        self.gguf_param_store.add_tensor_data_layout("jploski")  # qkv tensor transform
        self.gguf_param_store.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_param_store.add_feed_forward_length(4 * self.hparams["hidden_size"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_head_count(n_head)
        self.gguf_param_store.add_head_count_kv(n_head_kv)
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("num_hidden_layers")
        if block_count is None:
            block_count = self.hparams["n_layer"]  # old name

        n_head = self.hparams.get("num_attention_heads")
        if n_head is None:
            n_head = self.hparams["n_head"]  # old name

        n_head_kv = self.hparams.get("num_kv_heads")
        if n_head_kv is None:
            n_head_kv = self.hparams.get("n_head_kv", 1)  # old name

        head_dim = self.hparams["hidden_size"] // n_head
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # QKV tensor transform
            # The original query_key_value tensor contains n_head_kv "kv groups",
            # each consisting of n_head/n_head_kv query weights followed by one key
            # and one value weight (shared by all query heads in the kv group).
            # This layout makes it a big pain to work with in GGML.
            # So we rearrange them here,, so that we have n_head query weights
            # followed by n_head_kv key weights followed by n_head_kv value weights,
            # in contiguous fashion.
            # ref: https://github.com/jploski/ggml/blob/falcon40b/examples/falcon/convert-hf-to-ggml.py

            if "query_key_value" in name:
                qkv = data_torch.view(n_head_kv, n_head // n_head_kv + 2, head_dim, head_dim * n_head)
                q = qkv[:, :-2].reshape(n_head * head_dim, head_dim * n_head)
                k = qkv[:, [-2]].reshape(n_head_kv * head_dim, head_dim * n_head)
                v = qkv[:, [-1]].reshape(n_head_kv * head_dim, head_dim * n_head)
                data_torch = torch.cat((q, k, v)).reshape_as(data_torch)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)


class StarCoderModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_param_store.add_name("StarCoder")
        self.gguf_param_store.add_context_length(self.hparams["n_positions"])
        self.gguf_param_store.add_embedding_length(self.hparams["n_embd"])
        self.gguf_param_store.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_head_count(self.hparams["n_head"])
        self.gguf_param_store.add_head_count_kv(1)
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_file_type(self.ftype)


class RefactModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        block_count = self.hparams["n_layer"]

        self.gguf_param_store.add_name("Refact")
        # refact uses Alibi. So this is from config.json which might be used by training.
        self.gguf_param_store.add_context_length(self.hparams["n_positions"])
        self.gguf_param_store.add_embedding_length(self.hparams["n_embd"])

        self.gguf_param_store.add_feed_forward_length(ff_dim)
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_head_count(self.hparams["n_head"])
        self.gguf_param_store.add_head_count_kv(1)
        self.gguf_param_store.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_file_type(self.ftype)

    def write_tensors(self):
        hidden_dim = self.hparams["n_embd"]
        inner_dim = 4 * hidden_dim
        hidden_dim = int(2 * inner_dim / 3)
        multiple_of = 256
        ff_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        n_head = self.hparams["n_head"]
        n_head_kv = 1
        head_dim = self.hparams["n_embd"] // n_head
        block_count = self.hparams["n_layer"]

        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        tensors = dict(self.get_tensors())
        for i in range(block_count):
            if (w := tensors.get(f"transformer.h.{i}.attn.kv.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = w[:n_head_kv * head_dim]
                tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = w[n_head_kv * head_dim:]
                del tensors[f"transformer.h.{i}.attn.kv.weight"]
            if (w := tensors.get(f"transformer.h.{i}.attn.q.weight")) is not None:
                tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = w
                del tensors[f"transformer.h.{i}.attn.q.weight"]
            if (w := tensors.get(f"transformer.h.{i}.mlp.gate_up_proj.weight")) is not None:
                tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = w[:ff_dim]
                tensors[f"model.layers.{i}.mlp.up_proj.weight"] = w[ff_dim:]
                del tensors[f"transformer.h.{i}.mlp.gate_up_proj.weight"]

        for name, data_torch in tensors.items():
            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight",))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)


class PersimmonModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        head_count = self.hparams["num_attention_heads"]
        head_count_kv = head_count
        hidden_size = self.hparams["hidden_size"]

        self.gguf_param_store.add_name('persimmon-8b-chat')
        self.gguf_param_store.add_embedding_length(hidden_size)
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_param_store.add_rope_dimension_count(hidden_size // head_count)
        self.gguf_param_store.add_head_count(head_count)
        self.gguf_param_store.add_head_count_kv(head_count_kv)
        self.gguf_param_store.add_rope_freq_base(self.hparams["rope_theta"])
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_eps"])
        self.gguf_param_store.add_layer_norm_rms_eps(self.hparams["rms_norm_eps"])

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if name.endswith(".self_attention.rotary_emb.inv_freq"):
                continue
            old_dtype = data_torch.dtype
            # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
            data = data_torch.to(torch.float32).squeeze().numpy()
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()
            n_dims = len(data.shape)
            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_param_store.add_tensor(new_name, data)


class StableLMModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_context_length(hparams["max_position_embeddings"])
        self.gguf_param_store.add_embedding_length(hparams["hidden_size"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_param_store.add_rope_dimension_count(int(hparams["rope_pct"] * (hparams["hidden_size"] // hparams["num_attention_heads"])))
        self.gguf_param_store.add_head_count(hparams["num_attention_heads"])
        self.gguf_param_store.add_parallel_residual(hparams["use_parallel_residual"] if "use_parallel_residual" in hparams else True)
        self.gguf_param_store.add_layer_norm_eps(1e-5)


class MixtralModel(GGUFExportedModelDescriptor):
    pass


class QwenModel(GGUFExportedModelDescriptor):
    @staticmethod
    def token_bytes_to_string(b):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        byte_encoder = bytes_to_unicode()
        return ''.join([byte_encoder[ord(char)] for char in b.decode('latin-1')])

    @staticmethod
    def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
        parts = [bytes([b]) for b in token]
        while True:
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            assert min_idx is not None
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts

    def set_gguf_parameters(self):
        self.gguf_param_store.add_name("Qwen")
        self.gguf_param_store.add_context_length(self.hparams["max_position_embeddings"])
        self.gguf_param_store.add_block_count(self.hparams["num_hidden_layers"])
        self.gguf_param_store.add_embedding_length(self.hparams["hidden_size"])
        self.gguf_param_store.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_param_store.add_rope_freq_base(self.hparams["rotary_emb_base"])
        self.gguf_param_store.add_rope_dimension_count(self.hparams["hidden_size"] // self.hparams["num_attention_heads"])
        self.gguf_param_store.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_param_store.add_layer_norm_rms_eps(self.hparams["layer_norm_epsilon"])

    def write_tensors(self):
        block_count = self.hparams["num_hidden_layers"]
        model_kv = dict(self.get_tensors())
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)
        for name, data_torch in model_kv.items():
            # we don't need these
            if name.endswith(".rotary_emb.inv_freq"):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
            self.gguf_param_store.add_tensor(new_name, data)


class GPT2Model(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        self.gguf_param_store.add_name(self.model.name_or_path)
        self.gguf_param_store.add_block_count(self.hparams["n_layer"])
        self.gguf_param_store.add_context_length(self.hparams["n_ctx"])
        self.gguf_param_store.add_embedding_length(self.hparams["n_embd"])
        self.gguf_param_store.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_param_store.add_head_count(self.hparams["n_head"])
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_file_type(self.ftype)

    def write_tensors(self):
        block_count = self.hparams.get("n_layers", self.hparams.get("num_hidden_layers", self.hparams.get("n_layer")))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith((".attention.masked_bias", ".attention.bias", ".attention.rotary_emb.inv_freq", ".attn.bias")):
                continue

            if name.endswith((".c_attn.weight", ".c_proj.weight", ".c_fc.weight", ".c_proj.weight")):
                data_torch = data_torch.transpose(1, 0)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)

            # note: GPT2 output is tied to (same as) wte in original model
            if new_name == "token_embd.weight":
                print(f"output.weight, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")
                self.gguf_param_store.add_tensor("output.weight", data)


class Phi2Model(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]

        self.gguf_param_store.add_name("Phi2")
        self.gguf_param_store.add_context_length(self.hparams["n_positions"])
        self.gguf_param_store.add_embedding_length(self.hparams["n_embd"])
        self.gguf_param_store.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_head_count(self.hparams["n_head"])
        self.gguf_param_store.add_head_count_kv(self.hparams["n_head"])
        self.gguf_param_store.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_param_store.add_rope_dimension_count(self.hparams["rotary_dim"])
        self.gguf_param_store.add_file_type(self.ftype)
        self.gguf_param_store.add_add_bos_token(False)


class PlamoModel(GGUFExportedModelDescriptor):
    def set_gguf_parameters(self):
        hparams = self.hparams
        block_count = hparams["num_hidden_layers"]

        self.gguf_param_store.add_name("PLaMo")
        self.gguf_param_store.add_context_length(4096)  # not in config.json
        self.gguf_param_store.add_embedding_length(hparams["hidden_size"])
        self.gguf_param_store.add_feed_forward_length(hparams["intermediate_size"])
        self.gguf_param_store.add_block_count(block_count)
        self.gguf_param_store.add_head_count(hparams["num_attention_heads"])
        self.gguf_param_store.add_head_count_kv(5)  # hparams["num_key_value_heads"]) is wrong
        self.gguf_param_store.add_layer_norm_rms_eps(hparams["rms_norm_eps"])

    def shuffle_attn_q_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(8, 5, 128, 5120)
        data_torch = torch.permute(data_torch, (1, 0, 2, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def shuffle_attn_output_weight(self, data_torch):
        assert data_torch.size() == (5120, 5120)
        data_torch = data_torch.reshape(5120, 8, 5, 128)
        data_torch = torch.permute(data_torch, (0, 2, 1, 3))
        data_torch = torch.reshape(data_torch, (5120, 5120))
        return data_torch

    def write_tensors(self):
        block_count = self.hparams.get("num_layers", self.hparams.get("num_hidden_layers"))
        tensor_map = gguf.get_tensor_name_map(self.model_arch, block_count)

        for name, data_torch in self.get_tensors():
            if "self_attn.rotary_emb.inv_freq" in name:
                continue

            # map tensor names
            new_name = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
            if new_name is None:
                print(f"Can not map tensor {name!r}")
                sys.exit()

            # shuffle for broadcasting of gqa in ggml_mul_mat
            if new_name.endswith("attn_q.weight"):
                data_torch = self.shuffle_attn_q_weight(data_torch)
            elif new_name.endswith("attn_output.weight"):
                data_torch = self.shuffle_attn_output_weight(data_torch)

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            data = data_torch.squeeze().numpy()

            n_dims = len(data.shape)
            data_dtype = data.dtype

            # if f32 desired, convert any float16 to float32
            if self.ftype == 0 and data_dtype == np.float16:
                data = data.astype(np.float32)

            # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
            if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
                data = data.astype(np.float32)

            # if f16 desired, convert any float32 2-dim weight tensors to float16
            if self.ftype == 1 and data_dtype == np.float32 and name.endswith(".weight") and n_dims == 2:
                data = data.astype(np.float16)

            print(f"{new_name}, n_dims = {n_dims}, {old_dtype} --> {data.dtype}")

            self.gguf_param_store.add_tensor(new_name, data)


def get_gguf_params(model: PreTrainedModel) -> Dict[str, Any]:
    with torch.inference_mode():
        model_descriptor_class = GGUFExportedModelDescriptor.from_model_architecture(model.__class__.__name__)
        model_descriptor_instance = model_descriptor_class(model)
        model_descriptor_instance.set_gguf_parameters()
    return model_descriptor_instance.get_params_dict()
