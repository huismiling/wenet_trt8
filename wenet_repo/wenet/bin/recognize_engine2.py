from __future__ import print_function

import argparse
import copy
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

import multiprocessing
import numpy as np

import ctypes
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

from pathlib import Path
import time


try:
    from swig_decoders import map_batch, \
        ctc_beam_search_decoder_batch, \
        TrieVector, PathTrie
except ImportError:
    print('Please install ctc decoders first by refering to\n' +
          'https://github.com/Slyne/ctc_decoder.git')
    sys.exit(1)


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
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--encoder_plan', required=True, help='encoder onnx file')
    parser.add_argument('--decoder_plan', required=True, help='decoder onnx file')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='asr result file')
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
    parser.add_argument('--so',
                        default=None,
                        type=str,
                        help='plugin so file')
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
    soFileList = []
    for i in Path(args.so).iterdir():
        if i.suffix == '.so':
            soFileList.append(str(i))

    print(f'Plugin soFileList is\n{soFileList}')

    time_list = [] # (index,encoder_infer_time,decoder_infer_time),...

    cudart.cudaDeviceSynchronize()

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(str(soFile))

    with open(str(args.encoder_plan),'rb') as encoderF:
        encoder = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
    with open(str(args.decoder_plan), 'rb') as decoderF:
        decoder = trt.Runtime(logger).deserialize_cuda_engine(decoderF.read())

    _, encoder_stream = cudart.cudaStreamCreate()
    _, decoder_stream = cudart.cudaStreamCreate()
    encoder_context = encoder.create_execution_context()
    decoder_context = decoder.create_execution_context()

    # encoder_context.set_optimization_profile_async(0, encoder_stream)
    # decoder_context.set_optimization_profile_async(1, decoder_stream)

    encoder_nInput = np.sum([encoder.binding_is_input(i) for i in range(encoder.num_bindings)])
    encoder_nOutput = encoder.num_bindings - encoder_nInput

    decoder_nInput = np.sum([decoder.binding_is_input(i) for i in range(decoder.num_bindings)])
    decoder_nOutput = decoder.num_bindings - decoder_nInput

    print(f'Encoder 有 {encoder_nInput} 个输入')
    print(f'Encoder 有 {encoder_nOutput} 个输出')


    print(f'Decoder 有 {decoder_nInput} 个输入')
    print(f'Decoder 有 {decoder_nOutput} 个输出')

    Flag = True

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
        for index, batch in enumerate(test_data_loader):
            timelog = [index,None,None]
            keys, feats, _, feats_lengths, _ = batch
            feats, feats_lengths = feats.numpy(), feats_lengths.numpy()

            encoder_context.set_binding_shape(0, feats.shape)
            encoder_context.set_binding_shape(1, feats_lengths.shape)

            bufferH = []
            bufferH.append( feats.astype(np.float32).reshape(-1) )
            bufferH.append( feats_lengths.astype(np.int32).reshape(-1) )

            for i in range(encoder_nInput, encoder_nInput + encoder_nOutput):
                bufferH.append( np.empty(encoder_context.get_binding_shape(i), dtype=trt.nptype(encoder.get_binding_dtype(i))) )

            bufferD = []
            for i in range(encoder_nInput + encoder_nOutput):
                bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1] )

            for i in range(encoder_nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            if Flag:
                for i in range(10):
                    encoder_context.execute_v2(bufferD)
            # infer encoder
            t0 = time.time_ns()
            encoder_context.execute_v2(bufferD)
            t1 = time.time_ns()
            timelog[1] = (t1-t0)/1000/1000


            for i in range(encoder_nInput, encoder_nInput + encoder_nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            
            for i in range(encoder_nInput + encoder_nOutput):                
                cudart.cudaFree(bufferD[i])

            index_encoder_out = encoder.get_binding_index('encoder_out')
            index_encoder_out_lens = encoder.get_binding_index('encoder_out_lens')
            index_ctc_log_probs = encoder.get_binding_index('ctc_log_probs')
            index_beam_log_probs = encoder.get_binding_index('beam_log_probs')
            index_beam_log_probs_idx = encoder.get_binding_index('beam_log_probs_idx')

            encoder_out, encoder_out_lens, ctc_log_probs, \
                beam_log_probs, beam_log_probs_idx = bufferH[index_encoder_out],bufferH[index_encoder_out_lens],\
                bufferH[index_ctc_log_probs],bufferH[index_beam_log_probs],bufferH[index_beam_log_probs_idx]


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
                max_len = 62
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    cur_ctc_score = []
                    for hyp in hyps:
                        cur_ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                    ctc_score.append(cur_ctc_score)

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
                
                decoder_context.set_binding_shape(0, encoder_out.shape)
                decoder_context.set_binding_shape(1, encoder_out_lens.shape)
                decoder_context.set_binding_shape(2, hyps_pad_sos_eos.shape)
                decoder_context.set_binding_shape(3, hyps_lens_sos.shape)
                decoder_context.set_binding_shape(4, ctc_score.shape)

                bufferH = []
                bufferH.append( encoder_out.astype(np.float32).reshape(-1) )
                bufferH.append( encoder_out_lens.astype(np.int32).reshape(-1) )
                bufferH.append( hyps_pad_sos_eos.astype(np.int32).reshape(-1) )
                bufferH.append( hyps_lens_sos.astype(np.int32).reshape(-1) )        
                bufferH.append( ctc_score.astype(np.float32).reshape(-1) )   

                for i in range(decoder_nInput, decoder_nInput + decoder_nOutput):               
                    bufferH.append( np.empty(decoder_context.get_binding_shape(i), dtype=trt.nptype(decoder.get_binding_dtype(i))) )

                bufferD = []
                for i in range(decoder_nInput + decoder_nOutput):                
                    bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

                for i in range(decoder_nInput):
                    cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

                if Flag:
                    for i in range(10):
                        decoder_context.execute_v2(bufferD)
                    Flag = False
                # decoder infer
                t0 = time.time_ns()
                decoder_context.execute_v2(bufferD)
                t1 = time.time_ns()
                timelog[2] = (t1-t0)/1000/1000
                time_list.append(timelog)

                for i in range(decoder_nInput, decoder_nInput + decoder_nOutput):
                    cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

                for i in range(decoder_nInput + decoder_nOutput):                
                    cudart.cudaFree(bufferD[i])
                indexDecoderOut = decoder.get_binding_index('decoder_out')
                indexBestIndex = decoder.get_binding_index('best_index')

                best_index = bufferH[indexBestIndex]

                best_sents = []
                k = 0
                for idx in best_index:
                    cur_best_sent = all_hyps[k: k + beam_size][idx]
                    best_sents.append(cur_best_sent)
                    k += beam_size
                hyps = map_batch(best_sents, vocabulary, num_processes)            

            for i, key in enumerate(keys):
                content = hyps[i]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
    path = Path(args.result_file).parent / 'npys'/ f'engine2_{args.mode}'
    np.save(path,np.array(time_list))               

if __name__ == '__main__':
    main()
