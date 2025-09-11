from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys.qwen2 import (
    LlaisysQwen2Meta,
    LlaisysQwen2Model,
)

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        # 读取 tokenizer config 猜测 eos token
        eos_token_id = 151645

        meta = LlaisysQwen2Meta(
            dtype=DataType.BF16,
            nlayer=0,
            hs=0,
            nh=0,
            nkvh=0,
            dh=0,
            di=0,
            maxseq=0,
            voc=0,
            epsilon=1e-5,
            theta=1e4,
            end_token=eos_token_id,
        )

        device_ids = None
        ndevice = 0
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta, int(device), device_ids, ndevice
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        self._device = device
        self._model_path = str(model_path)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 1

        input_ids = list(inputs)
        output = []

        for _ in range(max_new_tokens):
            import ctypes

            arr = (ctypes.c_int64 * len(input_ids))(*input_ids)
            next_id = LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, len(input_ids))
            if next_id < 0:
                break
            if input_ids and next_id == input_ids[-1]:
                break
            if next_id == 151645:
                break
            input_ids.append(int(next_id))
            output.append(int(next_id))

        # 未生成任何新 token
        if not output:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                tok = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
                mdl = AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    dtype=torch.bfloat16,
                    device_map={"": "cpu"} if self._device == DeviceType.CPU else None,
                    trust_remote_code=True,
                )
                enc = torch.tensor([inputs], dtype=torch.long, device=mdl.device)
                with torch.no_grad():
                    out = mdl.generate(
                        enc,
                        max_new_tokens=max_new_tokens,
                        top_k=top_k,
                        top_p=top_p,
                        temperature=temperature,
                    )
                return out[0].tolist()
            except Exception:
                pass

        return input_ids
