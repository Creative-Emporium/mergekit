# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class LinearMergeTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo
    tokenizer_source: Optional[str] = None
    new_vocab_size: Optional[int] = None

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        if self.weight_info.name in ["tok_embeddings.weight", "lm_head.weight"] and self.new_vocab_size:
            base_tensor = tensors[self.base_model]
            return reshape_embedding_weights(base_tensor, self.new_vocab_size)

        keys = list(tensors.keys())
        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        rectify_embed_sizes(self.weight_info, tensors)

        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )

        tensors = torch.stack(tensors, dim=0)
        weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)
        while len(weights.shape) < len(tensors.shape):
            weights.unsqueeze_(-1)

        res = (weights * tensors).sum(dim=0)
        if self.normalize:
            res = res / weights.sum(dim=0)

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class LinearMerge(MergeMethod):
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
            ConfigParameterDef(name="tokenizer_source", required=False, default_value=None),
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
            tokenizer_source=parameters.get("tokenizer_source", None),
            new_vocab_size=128256 if parameters.get("tokenizer_source") == "NewerTokenizer" else None
        )

def reshape_embedding_weights(embedding_weight: torch.Tensor, new_vocab_size: int) -> torch.Tensor:
    current_vocab_size, embedding_dim = embedding_weight.shape
    if new_vocab_size > current_vocab_size:
        # Expand the embedding weight
        new_embedding_weight = torch.nn.Parameter(torch.zeros(new_vocab_size, embedding_dim))
        new_embedding_weight.data[:current_vocab_size, :] = embedding_weight.data
    else:
        # Truncate the embedding weight
        new_embedding_weight = torch.nn.Parameter(embedding_weight.data[:new_vocab_size, :])
    
    return new_embedding_weight

def reshape_model_for_new_vocab(
    model: torch.nn.Module,
    new_vocab_size: int,
    embedding_layer_name: str = "tok_embeddings.weight",
    lm_head_layer_name: str = "lm_head.weight"
) -> torch.nn.Module:
    # Reshape tok_embeddings
    tok_embeddings_weight = getattr(model, embedding_layer_name)
    new_tok_embeddings_weight = reshape_embedding_weights(tok_embeddings_weight, new_vocab_size)
    setattr(model, embedding_layer_name, new_tok_embeddings_weight)

    # Reshape lm_head
    lm_head_weight = getattr(model, lm_head_layer_name)
    new_lm_head_weight = reshape_embedding_weights(lm_head_weight, new_vocab_size)
    setattr(model, lm_head_layer_name, new_lm_head_weight)

    return model
