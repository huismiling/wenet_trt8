from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy


from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from export_onnx import Encoder,Decoder

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset,DataLoader


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


dataset = MyDataSet()

dataloader = DataLoader(dataset,
                       batch_size=1,
                       shuffle=True,
                       pin_memory=True,
                       num_workers=8)

with open("configs.json","r") as f:
    configs = json.load(f)
checkpoint = Path("20210204_conformer_exp/final.pt")

device = torch.device("cuda:0")
model = init_asr_model(configs)
load_checkpoint(model, str(checkpoint))
model = model.to(device)
model.eval()

encoder = Encoder(model.encoder, model.ctc, 10)
encoder = encoder.to(device)
encoder.eval()


backend = BackendType.Tensorrt
encoder = prepare_by_platform(encoder, backend)

enable_calibration(encoder)



for i, batch in enumerate(dataloader):
    feats, feats_lengths = batch
    feats = feats.to(device)
    feats_lengths = feats_lengths.to(device)
    out = encoder(feats, feats_lengths)

enable_quantization(encoder)

for i, batch in enumerate(dataloader):
    feats, feats_lengths = batch
    feats = feats.to(device)
    feats_lengths = feats_lengths.to(device)
    out = encoder(feats, feats_lengths)



print(encoder)