import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from pathlib import Path

class MyDataSet(Dataset):
    def __init__(self, path = Path('/home/ubuntu/Data/npys'),mode = 'encoder',device=None):
        self.device = device if device else torch.device('cuda:0')
        if mode == 'encoder':
            path = path / 'encoder'
            self.speech = path / "speech"
            self.speech_lengths = path / "speech_lengths"

            self.speech_list = [(i,data) for i,data in enumerate(self.speech.iterdir()) if data.is_file()]
            self.speech_lengths_list = [(i,data) for i,data in enumerate(self.speech_lengths.iterdir()) if data.is_file()]
            l1= len(self.speech_list)
            l2 = len(self.speech_lengths_list)
            assert l1 == l2
            self.length = l1
        elif mode == 'decoder':
            path = path / 'decoder'
            self.encoder_out = path / "encoder_out"
            self.encoder_out_lens = path / "encoder_out_lens"
            self.hyps_pad_sos_eos = path / "hyps_pad_sos_eos"
            self.hyps_lens_sos = path / "hyps_lens_sos"
            self.ctc_score = path / "ctc_score"

            self.encoder_out_list = [(i,data) for i,data in enumerate(self.encoder_out.iterdir()) if data.is_file()]
            self.encoder_out_lens_list = [(i,data) for i,data in enumerate(self.encoder_out_lens.iterdir()) if data.is_file()]
            self.hyps_pad_sos_eos_list = [(i, data) for i, data in enumerate(self.hyps_pad_sos_eos.iterdir()) if
                                          data.is_file()]
            self.hyps_lens_sos_list = [(i, data) for i, data in enumerate(self.hyps_lens_sos.iterdir()) if
                                          data.is_file()]
            self.ctc_score_list = [(i, data) for i, data in enumerate(self.ctc_score.iterdir()) if
                                          data.is_file()]
            l1 = len(self.encoder_out_list)
            l2 = len(self.encoder_out_lens_list)
            l3 = len(self.hyps_pad_sos_eos_list)
            l4 = len(self.hyps_lens_sos_list)
            l5 = len(self.ctc_score_list)
            assert l1 == l2 or l2 == l3 or l3 == l4 or l4 == l5
            self.length = l1
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.mode == 'encoder':
            speech = self.speech_list[item]
            speech_lengths = self.speech_lengths_list[item]
            speech = np.load(speech[1])
            speech_lengths = np.load(speech_lengths[1])
            speech = torch.from_numpy(speech).to(self.device)
            speech_lengths = torch.from_numpy(speech_lengths).to(self.device)
            sample = {
                'speech':speech,
                'speech_lengths':speech_lengths
            }
            return sample
        elif self.mode == 'decoder':
            encoder_out = self.encoder_out_list[item]
            encoder_out_lens = self.encoder_out_lens_list[item]
            hyps_pad_sos_eos = self.hyps_pad_sos_eos_list[item]
            hyps_lens_sos = self.hyps_lens_sos_list[item]
            ctc_score = self.ctc_score_list[item]
            encoder_out = np.load(encoder_out[1])
            encoder_out_lens = np.load(encoder_out_lens[1])
            hyps_pad_sos_eos = np.load(hyps_pad_sos_eos[1])
            hyps_lens_sos = np.load(hyps_lens_sos[1])
            ctc_score = np.load(ctc_score[1])
            encoder_out = torch.from_numpy(encoder_out).to(self.device)
            encoder_out_lens = torch.from_numpy(encoder_out_lens).to(self.device)
            hyps_pad_sos_eos = torch.from_numpy(hyps_pad_sos_eos).to(self.device)
            hyps_lens_sos = torch.from_numpy(hyps_lens_sos).to(self.device)
            ctc_score = torch.from_numpy(ctc_score).to(self.device)
            sample = {
                'encoder_out':encoder_out,
                'encoder_out_lens':encoder_out_lens,
                'hyps_pad_sos_eos':hyps_pad_sos_eos,
                'hyps_lens_sos':hyps_lens_sos,
                'ctc_score':ctc_score
            }
            return sample

if __name__ == '__main__':
    # dataset = MyDataSet()
    dataset = MyDataSet(mode='decoder')

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=8)
    for i in dataloader:
        print([v.shape for k,v in i.items()])
        break