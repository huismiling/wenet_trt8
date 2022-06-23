
from __future__ import print_function

import argparse
import os
import sys

import torch
import yaml
import logging

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.mask import make_pad_mask

from pathlib import Path

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


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
                speech_lengths: torch.Tensor, ):
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
        hyps_lens = hyps_lens_sos.view(B2, )
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
        return best_index




def export_offline_encoder(model, configs, args, logger, encoder_onnx_path):
    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(low=10, high=seq_len, size=(bz,), dtype=torch.int32)
    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()
    if args.dry:
        for _ in range(2):
            encoder(speech, speech_lens)
    if not args.jit:
        torch.onnx.export(encoder,
                          (speech, speech_lens),
                          encoder_onnx_path,
                          export_params=True,
                          opset_version=args.opset,
                          do_constant_folding=True,
                          input_names=['speech', 'speech_lengths'],
                          output_names=['encoder_out', 'encoder_out_lens',
                                        'ctc_log_probs',
                                        'beam_log_probs', 'beam_log_probs_idx'],
                          dynamic_axes={
                              'speech': {0: 'B', 1: 'T'},
                              'speech_lengths': {0: 'B'},
                              'encoder_out': {0: 'B', 1: 'T_OUT'},
                              'encoder_out_lens': {0: 'B'},
                              'ctc_log_probs': {0: 'B', 1: 'T_OUT'},
                              'beam_log_probs': {0: 'B', 1: 'T_OUT'},
                              'beam_log_probs_idx': {0: 'B', 1: 'T_OUT'},
                          },
                          verbose=False
                          )
    else:
        mod = torch.jit.trace(encoder, (speech, speech_lens))
        torch.jit.save(mod, (encoder_onnx_path.parent / 'jit' /encoder_onnx_path.stem/encoder_onnx_path.name ).with_suffix('.pt'))

    logger.info("export offline onnx encoder succeed!")
    onnx_config = {"beam_size": args.beam_size,
                   "reverse_weight": args.reverse_weight,
                   "ctc_weight": args.ctc_weight,
                   "fp16": False}
    return onnx_config





def export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path):
    bz, seq_len = 32, 100
    beam_size = args.beam_size
    decoder = Decoder(model.decoder,
                      model.ctc_weight,
                      model.reverse_weight,
                      beam_size)
    decoder.eval()

    hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))
    hyps_lens_sos = torch.randint(low=3, high=seq_len, size=(bz, beam_size),
                                  dtype=torch.int32)
    r_hyps_pad_sos_eos = torch.randint(low=3, high=1000, size=(bz, beam_size, seq_len))

    output_size = configs["encoder_conf"]["output_size"]
    encoder_out = torch.randn(bz, seq_len, output_size, dtype=torch.float32)
    encoder_out_lens = torch.randint(low=3, high=seq_len, size=(bz,), dtype=torch.int32)
    ctc_score = torch.randn(bz, beam_size, dtype=torch.float32)

    input_names = ['encoder_out', 'encoder_out_lens',
                   'hyps_pad_sos_eos', 'hyps_lens_sos',
                   'r_hyps_pad_sos_eos', 'ctc_score']
    if args.dry:
        for _ in range(2):
            decoder(encoder_out, encoder_out_lens,
                    hyps_pad_sos_eos, hyps_lens_sos,
                    r_hyps_pad_sos_eos, ctc_score)
    if not args.jit:
        torch.onnx.export(decoder,
                          (encoder_out, encoder_out_lens,
                           hyps_pad_sos_eos, hyps_lens_sos,
                           r_hyps_pad_sos_eos, ctc_score),
                          decoder_onnx_path,
                          export_params=True,
                          opset_version=args.opset,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=['best_index'],
                          dynamic_axes={'encoder_out': {0: 'B', 1: 'T'},
                                        'encoder_out_lens': {0: 'B'},
                                        'hyps_pad_sos_eos': {0: 'B', 2: 'T2'},
                                        'hyps_lens_sos': {0: 'B'},
                                        'r_hyps_pad_sos_eos': {0: 'B', 2: 'T2'},
                                        'ctc_score': {0: 'B'},
                                        'best_index': {0: 'B'},
                                        },
                          verbose=False
                          )
    else:
        mod = torch.jit.trace(decoder, (encoder_out, encoder_out_lens,
                           hyps_pad_sos_eos, hyps_lens_sos,
                           r_hyps_pad_sos_eos, ctc_score))
        torch.jit.save(mod, (decoder_onnx_path.parent / 'jit' /decoder_onnx_path.stem /encoder_onnx_path.name).with_suffix('.pt'))


    logger.info("export to onnx decoder succeed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export x86_gpu model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--cmvn_file', required=False, default='', type=str,
                        help='global_cmvn file, default path is in config file')
    parser.add_argument('--reverse_weight', default=-1.0, type=float,
                        required=False,
                        help='reverse weight for bitransformer,' +
                             'default value is in config file')
    parser.add_argument('--ctc_weight', default=-1.0, type=float,
                        required=False,
                        help='ctc weight, default value is in config file')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_onnx_dir',
                        default="onnx_model",
                        help='output onnx encoder and decoder directory')
    parser.add_argument('--decoding_chunk_size',
                        default=16,
                        type=int,
                        required=False,
                        help='the decoding chunk size, <=0 is not supported')
    parser.add_argument('--num_decoding_left_chunks',
                        default=5,
                        type=int,
                        required=False,
                        help="number of left chunks, <= 0 is not supported")
    parser.add_argument('--dry',
                        action='store_false',
                        help="whether to dry run")
    parser.add_argument('--jit',
                        action='store_true',
                        help="whether to export jit")
    parser.add_argument('--opset',
                        default=13,
                        type=int,
                        required=False,
                        help='onnx opset for torch export')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    configs = {'accum_grad': 4, 'cmvn_file': '/home/ubuntu/study/wenet/work_dir/20210204_conformer_exp/global_cmvn',
               'dataset_conf': {'batch_conf': {'batch_size': 16, 'batch_type': 'static'},
                                'fbank_conf': {'dither': 0.1, 'frame_length': 25, 'frame_shift': 10,
                                               'num_mel_bins': 80},
                                'filter_conf': {'max_length': 40960, 'min_length': 0, 'token_max_length': 200,
                                                'token_min_length': 1}, 'resample_conf': {'resample_rate': 16000},
                                'shuffle': True, 'shuffle_conf': {'shuffle_size': 1500}, 'sort': True,
                                'sort_conf': {'sort_size': 500}, 'spec_aug': True,
                                'spec_aug_conf': {'max_f': 10, 'max_t': 50, 'num_f_mask': 2, 'num_t_mask': 2},
                                'speed_perturb': True}, 'decoder': 'transformer',
               'decoder_conf': {'attention_heads': 4, 'dropout_rate': 0.1, 'linear_units': 2048, 'num_blocks': 6,
                                'positional_dropout_rate': 0.1, 'self_attention_dropout_rate': 0.0,
                                'src_attention_dropout_rate': 0.0}, 'encoder': 'conformer',
               'encoder_conf': {'activation_type': 'swish', 'attention_dropout_rate': 0.0, 'attention_heads': 4,
                                'cnn_module_kernel': 15, 'dropout_rate': 0.1, 'input_layer': 'conv2d',
                                'linear_units': 2048, 'normalize_before': True, 'num_blocks': 12, 'output_size': 256,
                                'pos_enc_layer_type': 'rel_pos', 'positional_dropout_rate': 0.1,
                                'selfattention_layer_type': 'rel_selfattn', 'use_cnn_module': True,
                                'use_dynamic_chunk': False}, 'grad_clip': 5, 'input_dim': 80, 'is_json_cmvn': True,
               'log_interval': 100, 'max_epoch': 240,
               'model_conf': {'ctc_weight': 0.3, 'length_normalized_loss': False, 'lsm_weight': 0.1}, 'optim': 'adam',
               'optim_conf': {'lr': 0.002}, 'output_dim': 4233, 'scheduler': 'warmuplr',
               'scheduler_conf': {'warmup_steps': 25000}}

    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    out_path = Path(args.output_onnx_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True,exist_ok=True)

    encoder_onnx_path = out_path /  'encoder.onnx'

    onnx_config = export_offline_encoder(model, configs, args, logger, encoder_onnx_path)

    decoder_onnx_path = out_path /  'decoder.onnx'
    export_rescoring_decoder(model, configs, args, logger, decoder_onnx_path)



    config_dir = os.path.join(args.output_onnx_dir, "config.yaml")
    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
