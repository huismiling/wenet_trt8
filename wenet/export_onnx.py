from __future__ import print_function

import argparse
import torch
import yaml
import logging

from pathlib import Path
from torch import quantization

from wenet.transformer.asr_model import init_asr_model
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.mask import make_pad_mask
from wenet.utils.checkpoint import load_checkpoint

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
        B,T,_ = speech.shape
        speech_lengths = speech_lengths[:B]
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
        return best_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', default='20210204_conformer_exp/train.yaml', help='config file')
    parser.add_argument('--checkpoint', default='20210204_conformer_exp/final.pt', help='checkpoint model')
    parser.add_argument('--cmvn_file', default='20210204_conformer_exp/global_cmvn', type=str,
                        help='global_cmvn file, default path is in config file')
    parser.add_argument('--reverse_weight', default=-1.0, type=float,
                        required=False,
                        help='reverse weight for bitransformer,' +
                        'default value is in config file')
    parser.add_argument('--ctc_weight', default=0.3, type=float,
                        required=False,
                        help='ctc weight, default value is in config file')
    parser.add_argument('--beam_size', default=10, type=int, required=False,
                        help="beam size would be ctc output size")
    parser.add_argument('--output_dir',
                        default="./out/",
                        help='output encoder and decoder directory')
    parser.add_argument('--torchscript',
                        action='store_true',
                        help='whether to export torchscript model, default false')
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)
    
    config = Path(args.config)
    cmvn_file = Path(args.cmvn_file)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True,exist_ok=True)

    with open(config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if cmvn_file and cmvn_file.exists():
        configs['cmvn_file'] = cmvn_file
    if args.reverse_weight != -1.0 and 'reverse_weight' in configs['model_conf']:
        configs['model_conf']['reverse_weight'] = args.reverse_weight
        print("Update reverse weight to", args.reverse_weight)
    if args.ctc_weight != -1:
        print("Update ctc weight to ", args.ctc_weight)
        configs['model_conf']['ctc_weight'] = args.ctc_weight
    configs["encoder_conf"]["use_dynamic_chunk"] = False
    model = init_asr_model(configs)
    load_checkpoint(model, str(checkpoint))
    model.eval()
    bz = 32
    seq_len = 100
    beam_size = args.beam_size
    feature_size = configs["input_dim"]

    encoder = Encoder(model.encoder, model.ctc, beam_size)
    encoder.eval()

    speech = torch.randn(bz, seq_len, feature_size, dtype=torch.float32)
    speech_lens = torch.randint(low=10, high=seq_len, size=(bz,), dtype=torch.int32)

    torch.onnx.export(encoder,
                      (speech, speech_lens),
                      output_dir / 'encoder.onnx',
                      export_params=True,
                      opset_version=13,
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
    logger.info("export to onnx encoder succeed!")

    if args.torchscript:
        mod = torch.jit.trace(encoder, (speech, speech_lens))
        torch.jit.save(mod, output_dir / 'encoder.pt')
        logger.info("export to torchscript encoder succeed!")

    decoder = Decoder(
        model.decoder,
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

    torch.onnx.export(decoder,
                      (encoder_out, encoder_out_lens,
                       hyps_pad_sos_eos, hyps_lens_sos,
                       r_hyps_pad_sos_eos, ctc_score),
                      output_dir / "decoder.onnx",
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names=['encoder_out', 'encoder_out_lens',
                                   'hyps_pad_sos_eos', 'hyps_lens_sos',
                                   'r_hyps_pad_sos_eos', 'ctc_score'],
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

    logger.info("export to onnx decoder succeed!")

    if args.torchscript:
        mod = torch.jit.trace(decoder,  (encoder_out, encoder_out_lens,
        hyps_pad_sos_eos, hyps_lens_sos,
        r_hyps_pad_sos_eos, ctc_score))
        torch.jit.save(mod, output_dir / 'decoder.pt')
        logger.info("export to torchscript decoder succeed!")

    # dump configurations
    onnx_config = {"beam_size": args.beam_size,
                   "reverse_weight": args.reverse_weight,
                   "ctc_weight": args.ctc_weight,
                   "fp16": False}

    config_dir = output_dir / "config.yaml"

    with open(config_dir, "w") as out:
        yaml.dump(onnx_config, out)
