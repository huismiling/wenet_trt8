# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is for testing exported onnx encoder and decoder from
export_onnx_gpu.py. The exported onnx models only support batch offline ASR inference.
It requires a python wrapped c++ ctc decoder.
Please install it by following:
https://github.com/Slyne/ctc_decoder.git
"""
from __future__ import print_function

import argparse
import copy
import json
import logging
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.utils.common import IGNORE_ID
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.mask import make_pad_mask

import multiprocessing
import numpy as np

try:
    from swig_decoders import map_batch, \
        ctc_beam_search_decoder_batch, \
        TrieVector, PathTrie
except ImportError:
    print('Please install ctc decoders first by refering to\n' +
          'https://github.com/Slyne/ctc_decoder.git')
    sys.exit(1)

class Encoder(torch.nn.Module):
    def __init__(self,
                 encoder: BaseEncoder,
                 ctc: CTC,
                 beam_size: int = 10):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc
        self.beam_size = beam_size

    def forward(self, speech: torch.Tensor,
                speech_lengths: torch.Tensor,):
        """Encoder
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        Returns:
            encoder_out: B x T x F
            encoder_out_lens: B
            ctc_log_probs: B x T x V
            beam_log_probs: B x T x beam_size
            beam_log_probs_idx: B x T x beam_size
        """
        encoder_out, encoder_mask = self.encoder(speech,
                                                 speech_lengths,
                                                 -1, -1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        encoder_out_lens = encoder_out_lens.int()
        beam_log_probs, beam_log_probs_idx = torch.topk(
            ctc_log_probs, self.beam_size, dim=2)
        return encoder_out, encoder_out_lens, ctc_log_probs, \
            beam_log_probs, beam_log_probs_idx

class Decoder(torch.nn.Module):
    def __init__(self,
                 decoder: TransformerDecoder,
                 ctc_weight: float = 0.5,
                 reverse_weight: float = 0.0,
                 beam_size: int = 10):
        super().__init__()
        self.decoder = decoder
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.beam_size = beam_size

    def forward(self,
                encoder_out: torch.Tensor,
                encoder_lens: torch.Tensor,
                hyps_pad_sos_eos: torch.Tensor,
                hyps_lens_sos: torch.Tensor,
                r_hyps_pad_sos_eos: torch.Tensor,
                ctc_score: torch.Tensor):
        """Encoder
        Args:
            encoder_out: B x T x F
            encoder_lens: B
            hyps_pad_sos_eos: B x beam x (T2+1),
                        hyps with sos & eos and padded by ignore id
            hyps_lens_sos: B x beam, length for each hyp with sos
            r_hyps_pad_sos_eos: B x beam x (T2+1),
                    reversed hyps with sos & eos and padded by ignore id
            ctc_score: B x beam, ctc score for each hyp
        Returns:
            decoder_out: B x beam x T2 x V
            r_decoder_out: B x beam x T2 x V
            best_index: B
        """
        B, T, F = encoder_out.shape
        bz = self.beam_size
        B2 = B * bz
        encoder_out = encoder_out.repeat(1, bz, 1).view(B2, T, F)
        encoder_mask = ~make_pad_mask(encoder_lens, T).unsqueeze(1)
        encoder_mask = encoder_mask.repeat(1, bz, 1).view(B2, 1, T)
        T2 = hyps_pad_sos_eos.shape[2] - 1
        hyps_pad = hyps_pad_sos_eos.view(B2, T2 + 1)
        hyps_lens = hyps_lens_sos.view(B2,)
        hyps_pad_sos = hyps_pad[:, :-1].contiguous()
        hyps_pad_eos = hyps_pad[:, 1:].contiguous()

        r_hyps_pad = r_hyps_pad_sos_eos.view(B2, T2 + 1)
        r_hyps_pad_sos = r_hyps_pad[:, :-1].contiguous()
        r_hyps_pad_eos = r_hyps_pad[:, 1:].contiguous()

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad_sos, hyps_lens, r_hyps_pad_sos,
            self.reverse_weight)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        V = decoder_out.shape[-1]
        decoder_out = decoder_out.view(B2, T2, V)
        mask = ~make_pad_mask(hyps_lens, T2)  # B2 x T2
        # mask index, remove ignore id
        index = torch.unsqueeze(hyps_pad_eos * mask, 2)
        score = decoder_out.gather(2, index).squeeze(2)  # B2 X T2
        # mask padded part
        score = score * mask
        decoder_out = decoder_out.view(B, bz, T2, V)
        if self.reverse_weight > 0:
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
            r_decoder_out = r_decoder_out.view(B2, T2, V)
            index = torch.unsqueeze(r_hyps_pad_eos * mask, 2)
            r_score = r_decoder_out.gather(2, index).squeeze(2)
            r_score = r_score * mask
            score = score * (1 - self.reverse_weight) + self.reverse_weight * r_score
            r_decoder_out = r_decoder_out.view(B, bz, T2, V)
        score = torch.sum(score, axis=1)  # B2
        score = torch.reshape(score, (B, bz)) + self.ctc_weight * ctc_score
        best_index = torch.argmax(score, dim=1)
        return decoder_out,best_index

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--checkpoint', required=True, help='encoder onnx file')
    parser.add_argument('--json', required=True, help='init model json file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beamsize')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 'ctc_prefix_beam_search',
                            'attention_rescoring'],
                        default='attention_rescoring',
                        help='decoding mode')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    reverse_weight = configs["model_conf"].get("reverse_weight", 0.0)
    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}')
    with open(args.json,'r') as f:
        init_config = json.load(f)
    model = init_asr_model(init_config)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    encoder = Encoder(model.encoder, model.ctc, args.beam_size)
    encoder.eval()

    decoder = Decoder(model.decoder,
                      model.ctc_weight,
                      model.reverse_weight,
                      args.beam_size)
    decoder.eval()


    if use_cuda:
        encoder.to(device)
        decoder.to(device)

    if args.fp16:
        encoder.half()
        decoder.half()


    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1

    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for _, batch in enumerate(test_data_loader):
            keys, feats, _, feats_lengths, _ = batch
            feats, feats_lengths = feats.to(device), feats_lengths.to(device)

            if args.fp16:
                feats = feats.half()

            encoder_outputs = encoder(feats,feats_lengths)

            encoder_out, encoder_out_lens, ctc_log_probs, \
                beam_log_probs, beam_log_probs_idx = encoder_outputs

            beam_size = beam_log_probs.shape[-1]
            batch_size = beam_log_probs.shape[0]

            num_processes = min(multiprocessing.cpu_count(), batch_size)

            if args.mode == 'ctc_greedy_search':
                if beam_size != 1:
                    log_probs_idx = beam_log_probs_idx[:, :, 0]
                batch_sents = []
                for idx, seq in enumerate(log_probs_idx):
                    batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
                hyps = map_batch(batch_sents, vocabulary, num_processes,
                                 True, 0)
            elif args.mode in ('ctc_prefix_beam_search', "attention_rescoring"):
                batch_log_probs_seq_list = beam_log_probs.tolist()
                batch_log_probs_idx_list = beam_log_probs_idx.tolist()
                batch_len_list = encoder_out_lens.tolist()
                batch_log_probs_seq = []
                batch_log_probs_ids = []
                batch_start = []  # only effective in streaming deployment
                batch_root = TrieVector()
                root_dict = {}
                for i in range(len(batch_len_list)):
                    num_sent = batch_len_list[i]
                    batch_log_probs_seq.append(
                        batch_log_probs_seq_list[i][0:num_sent])
                    batch_log_probs_ids.append(
                        batch_log_probs_idx_list[i][0:num_sent])
                    root_dict[i] = PathTrie()
                    batch_root.append(root_dict[i])
                    batch_start.append(True)
                score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                           batch_log_probs_ids,
                                                           batch_root,
                                                           batch_start,
                                                           beam_size,
                                                           num_processes,
                                                           0, -2, 0.99999)
                if args.mode == 'ctc_prefix_beam_search':
                    hyps = []
                    for cand_hyps in score_hyps:
                        hyps.append(cand_hyps[0][1])
                    hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
            if args.mode == 'attention_rescoring':
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    cur_ctc_score = []
                    for hyp in hyps:
                        cur_ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) > max_len:
                            max_len = len(hyp[1])
                    ctc_score.append(cur_ctc_score)
                if args.fp16:
                    ctc_score = np.array(ctc_score, dtype=np.float16)
                else:
                    ctc_score = np.array(ctc_score, dtype=np.float32)
                hyps_pad_sos_eos = np.ones(
                    (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
                r_hyps_pad_sos_eos = np.ones(
                    (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
                hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                k = 0
                for i in range(batch_size):
                    for j in range(beam_size):
                        cand = all_hyps[k]
                        l = len(cand) + 2
                        hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                        r_hyps_pad_sos_eos[i][j][0:l] = [sos] + cand[::-1] + [eos]
                        hyps_lens_sos[i][j] = len(cand) + 1
                        k += 1

                # if reverse_weight > 0:
                #     r_hyps_pad_sos_eos_name = decoder_ort_session.get_inputs()[4].name
                #     decoder_ort_inputs[r_hyps_pad_sos_eos_name] = r_hyps_pad_sos_eos
                hyps_pad_sos_eos = torch.from_numpy(hyps_pad_sos_eos).to(device)
                hyps_lens_sos = torch.from_numpy(hyps_lens_sos).to(device)
                r_hyps_pad_sos_eos = torch.from_numpy(r_hyps_pad_sos_eos).to(device)
                ctc_score = torch.from_numpy(ctc_score).to(device)
                _,best_index = decoder(encoder_out, encoder_out_lens,
                        hyps_pad_sos_eos, hyps_lens_sos,
                        r_hyps_pad_sos_eos, ctc_score)
                best_sents = []
                k = 0
                print(best_index)
                for idx in best_index:
                    cur_best_sent = all_hyps[k: k + beam_size][idx]
                    best_sents.append(cur_best_sent)
                    k += beam_size
                hyps = map_batch(best_sents, vocabulary, num_processes)

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))

if __name__ == '__main__':
    main()
