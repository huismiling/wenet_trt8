from typing import Iterable
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

from ppq import BaseGraph, QuantizationSettingFactory, TargetPlatform
from ppq.api import export_ppq_graph, quantize_torch_model

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from export_onnx import Encoder,Decoder

class MyDataSet(Dataset):
    def __init__(self, path = Path("/home/ubuntu/study/Down/npDataSets"),gt = False):
        self.speech = path / "speech"
        self.speech_lengths = path / "speech_lengths"
        self.target = path / "target"
        self.target_lengths = path / "target_lengths"
        self.gt = gt
        self.speech_list = [(i,data) for i,data in enumerate(self.speech.iterdir()) if data.is_file()]
        self.speech_lengths_list = [(i,data) for i,data in enumerate(self.speech_lengths.iterdir()) if data.is_file()]
        self.target_list = [(i,data) for i,data in enumerate(self.target.iterdir()) if data.is_file()]
        self.target_lengths_list = [(i,data) for i,data in enumerate(self.target_lengths.iterdir()) if data.is_file()]
        l1= len(self.speech_list)
        l2 = len(self.speech_lengths_list)
        l3 = len(self.target_list)
        l4 = len(self.target_lengths_list)
        tmp = [l1, l2, l3, l4]
        assert len(set(tmp)) == 1
        self.length = l1

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        speech = self.speech_list[item]
        speech_lengths = self.speech_lengths_list[item]
        speech = np.load(speech[1])
        speech_lengths = np.load(speech_lengths[1])
        speech = torch.from_numpy(speech)
        speech_lengths = torch.from_numpy(speech_lengths)
        target, target_lengths = None,None
        if self.gt:
            target = self.target_list[item]
            target_lengths = self.target_lengths_list[item]
            target = np.load(target[1])
            target_lengths = np.load(target_lengths[1])
            target = torch.from_numpy(target)
            target_lengths = torch.from_numpy(target_lengths)
        return (speech, speech_lengths) if not self.gt else (speech,speech_lengths, target ,target_lengths)



BATCHSIZE = 16
DEVICE = 'cuda' # only cuda is fully tested :(  For other executing device there might be bugs.
PLATFORM = TargetPlatform.PPL_CUDA_INT8  # identify a target platform for your network.

with open("configs.json","r") as f:
    configs = json.load(f)
checkpoint = Path("20210204_conformer_exp/final.pt")

model = init_asr_model(configs)
load_checkpoint(model, str(checkpoint))
model.eval()
model = model.to(DEVICE)


encoder = Encoder(model.encoder, model.ctc, 10)
encoder = encoder.to(DEVICE)
encoder.eval()

speech = torch.randn(BATCHSIZE, 100, 80, dtype=torch.float32).to(DEVICE)
speech_lens = torch.randint(low=10, high=100, size=(BATCHSIZE,), dtype=torch.int32).to(DEVICE)


dataset = MyDataSet()

dataloader = DataLoader(dataset,
                       batch_size=1,
                       shuffle=True,
                       pin_memory=True,
                       num_workers=8)

def load_calibration_dataset() -> Iterable:
    return dataloader



def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)



# create a setting for quantizing your network with PPL CUDA.
quant_setting = QuantizationSettingFactory.pplcuda_setting()
quant_setting.equalization = True # use layerwise equalization algorithm.
quant_setting.dispatcher   = 'conservative' # dispatch this network in conservertive way.

# Load training data for creating a calibration dataloader.
calibration_dataset = load_calibration_dataset()

calibration_dataloader = DataLoader(
    dataset=calibration_dataset,
    batch_size=None, pin_memory=True,num_workers=8,prefetch_factor=100)
tmp = {'speech':speech,"speech_lens":speech_lens}

# quantize your model.
quantized = quantize_torch_model(
    model=encoder, calib_dataloader=calibration_dataloader,
    calib_steps=32, input_dtype= (torch.float32,torch.int32) ,
    inputs=(speech,speech_lens),input_shape=([BATCHSIZE, 100, 80],[BATCHSIZE,]),
    setting=quant_setting, collate_fn=collate_fn, platform=PLATFORM,
    onnx_export_file='Output/onnx.model', device=DEVICE, verbose=0)

# Quantization Result is a PPQ BaseGraph instance.
assert isinstance(quantized, BaseGraph)

# export quantized graph.
export_ppq_graph(graph=quantized, platform=PLATFORM,
                 graph_save_to='Output/quantized(onnx).onnx',
                 config_save_to='Output/quantized(onnx).json')
