import os
import sys
import torch
import pickle
from tqdm import tqdm

# Compatibility shim for pytorchvideo==0.1.5 on newer torchvision.
# pytorchvideo imports torchvision.transforms.functional_tensor, which was moved
# to torchvision.transforms._functional_tensor in torchvision>=0.17.
try:
    import torchvision.transforms.functional_tensor  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms._functional_tensor as _functional_tensor  # type: ignore
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ImageBindModel, ModalityType


def encode_video_segments(video_paths, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.VISION: data.load_and_transform_video_data(video_paths, device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.VISION]
    embeddings = embeddings.cpu()
    return embeddings

def encode_string_query(query:str, embedder: ImageBindModel):
    device = next(embedder.parameters()).device
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text([query], device),
    }
    with torch.no_grad():
        embeddings = embedder(inputs)[ModalityType.TEXT]
    embeddings = embeddings.cpu()
    return embeddings
