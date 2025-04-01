# federated/utils.py

import torch
import numpy as np

def get_parameters(model):
    """PyTorch 모델의 파라미터를 NumPy 배열로 변환하여 리스트로 반환"""
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_parameters(model, parameters):
    """NumPy 배열 리스트를 PyTorch 모델 파라미터로 설정"""
    state_dict = model.state_dict()
    new_state_dict = {}
    for key, val in zip(state_dict.keys(), parameters):
        new_state_dict[key] = torch.tensor(val)
    model.load_state_dict(new_state_dict)
