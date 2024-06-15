from typing import Any, Dict, List, Optional, Tuple

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.tokenizer import build_tokenizer, TokenizerInfo


class LinearMergeTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo
    tokenizer_source: str
    trust_remote_code: bool = False

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs) -> torch.Tensor:
        keys = list(tensors.keys())
        tensors_list = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        unique_shapes = set(t.shape for t in tensors_list)
        if len(unique_shapes) != 1:
            raise RuntimeError(f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}")

        tensors = torch.stack(tensors_list, dim=0)
        weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)
        while len(weights.shape) < len(tensors.shape):
            weights.unsqueeze_(-1)

        res = (weights * tensors).sum(dim=0)
        if self.normalize:
            res = res / weights.sum(dim=0)

        # Tokenizer adjustment
        tokenizer, permutations = build_tokenizer(
            None,
            list(tensors.keys()),
            self.tokenizer_source,
            self.trust_remote_code,
        )
        vocab_out = tokenizer.get_vocab()
        new_vocab_size = len(vocab_out)
        
        if self.weight_info.name == "tok_embeddings.weight":
            current_shape = res.shape
            new_shape = (new_vocab_size, current_shape[1])
            res = torch.nn.functional.pad(res, (0, 0, 0, new_vocab_size - current_shape[0]))
        elif self.weight_info.name == "lm_head.weight":
            current_shape = res.shape
            new_shape = (new_vocab_size, current_shape[1])
            res = torch.nn.functional.pad(res, (0, new_vocab_size - current_shape[0], 0, 0))

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class LinearMerge(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
            ConfigParameterDef(name="tokenizer_source", required=True),
            ConfigParameterDef(name="trust_remote_code", required=False, default_value=False),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return LinearMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            weight_info=output_weight,
            tokenizer_source=parameters["tokenizer_source"],
            trust_remote_code=parameters.get("trust_remote_code", False),
        )
