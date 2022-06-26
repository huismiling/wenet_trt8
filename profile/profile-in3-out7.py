#!/usr/bin/python

import os
import sys
import ctypes
import numpy as np
from glob import glob 
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt

dataFilePath = "../datasets/data/"
planFilePath   = "../"
encoderPlanFile  = planFilePath + "encoder.plan"
encoderScoreFile = planFilePath + "log/encoderScore.txt"
decoderPlanFile  = planFilePath + "decoder.plan"
decoderScoreFile = planFilePath + "log/decoderScore.txt"
soFileList = glob(planFilePath + "*.so")
soFileList = ["../FasterTransformer_wenet/build/lib/libwenet_plugin.so"]

tableHead = \
"""
bs: Batch Size
sl: Sequence Length
lt: Latency (ms)
tp: throughput (word/s)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
a1: maximum of absolute difference of output 1
r1: median of relative difference of output 1
----+----+--------+---------+---------+---------+---------+---------+-------------
  bs|  sl|      lt|       tp|       a0|       r0|       a1|       r1| output check
----+----+--------+---------+---------+---------+---------+---------+-------------
"""

def printArrayInfo(x, description=""):
    print( '%s: %s\n  Mean=%.5e,SumAbs=%.5e,Var=%.5e,Max=%.5f,Min=%.5f,SAD=%.5e'%( \
        description,str(x.shape),np.mean(x),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print("\t", x.reshape(-1)[:10])

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    #print("check:",res,diff0,diff1)
    return res,diff0,diff1

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

def gen_decoder_mask(q_lens, kv_lens, q_max_len, kv_max_len):
    batch_size = len(kv_lens)
    self_mask = []
    cross_mask = []
    for itb in range(batch_size):
        kv_mask = np.zeros((q_max_len, kv_max_len), dtype=np.float32)
        for itl in range(q_max_len):
            kv_mask[itl, :kv_lens[itb]] = 1
        cross_mask.append(np.repeat(kv_mask[np.newaxis, ], 10, 0))
        
    for itb in range(batch_size*10):
        q_mask = np.zeros((q_max_len, q_max_len), dtype=np.float32)
        for itl in range(q_max_len):
            q_mask[itl, :min(itl+1, q_lens[itb])] = 1
        self_mask.append(q_mask[np.newaxis, ])
        
    self_mask = np.concatenate(self_mask)
    cross_mask = np.concatenate(cross_mask)
    return self_mask, cross_mask

def gen_encoder_mask(q_lens, q_max_len):
    batch_size = len(q_lens)
    self_mask = []
    for itb in range(batch_size):
        q_mask = np.zeros((q_max_len, q_max_len), dtype=np.float32)
        for itl in range(q_max_len):
            q_mask[itl, :q_lens[itb]] = 1
        self_mask.append(q_mask[np.newaxis, ])
    self_mask = np.concatenate(self_mask)
    return self_mask

#-------------------------------------------------------------------------------
def testEncoder():
    print("Test Encoder Part!")

    with open(encoderScoreFile, 'w') as f:

        if os.path.isfile(encoderPlanFile):
            with open(encoderPlanFile, 'rb') as encoderF:
                engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if engine is None:
                print("Failed loading %s"%encoderPlanFile)
                return
            print("Succeeded loading %s"%encoderPlanFile)
        else:
            print("Failed finding %s"%encoderPlanFile)
            return

        nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
        nOutput = engine.num_bindings - nInput
        context = engine.create_execution_context()
            
        print(tableHead)  # for standard output

        for ioFile in sorted(glob(dataFilePath + "./encoder-*.npz")):
            ioData = np.load(ioFile)
            speech = ioData['speech']
            speech_lengths = ioData['speech_lengths']
            batchSize, sequenceLength, _ = speech.shape
            if batchSize > 16 or sequenceLength > 1024:
                continue
            
            attn_mask = gen_encoder_mask(((speech_lengths+3)//4).tolist(), (speech.shape[1]-4)//4)
            # print(speech.shape, speech_lengths)
            context.set_binding_shape(0, speech.shape)
            context.set_binding_shape(1, speech_lengths.shape)
            context.set_binding_shape(2, attn_mask.shape)
            #for i in range(nInput + nOutput):
            #    print("Input ->" if engine.binding_is_input(i) else "Output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_dtype(i), engine.get_binding_name(i))
            #print("Finish all input binding: %s"%context.all_binding_shapes_specified)
            
            bufferH = []
            bufferH.append( speech.astype(np.float32).reshape(-1) )
            bufferH.append( speech_lengths.astype(np.int32).reshape(-1) )
            bufferH.append( attn_mask.astype(np.float32).reshape(-1) )
            for i in range(nInput, nInput + nOutput):                
                bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

            bufferD = []
            for i in range(nInput + nOutput):                
                bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            # warm up
            for i in range(10):
                context.execute_v2(bufferD)

            # test infernece time
            t0 = time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            t1 = time_ns()
            timePerInference = (t1-t0)/1000/1000/30

            indexEncoderOut = engine.get_binding_index('encoder_out')
            indexEncoderOutLens = engine.get_binding_index('encoder_out_lens')
            # index725 = engine.get_binding_index('725')
            # index742 = engine.get_binding_index('742')
            
            check0 = check(bufferH[indexEncoderOut],ioData['encoder_out'],True,5e-5)
            check1 = check(bufferH[indexEncoderOutLens],np.sum(ioData['encoder_out_lens'].astype(np.int32),axis=2)[:,0],True)
            # np.savetxt("tmp.txt", bufferH[index725].reshape(-1))
            # np.savetxt("tmp1.txt", bufferH[index742].reshape(-1))
            # exit(0)

            string = "%4d,%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e, %s"%(batchSize,
                                                                        sequenceLength,
                                                                        timePerInference,
                                                                        batchSize*sequenceLength/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check1[1],
                                                                        check1[2],
                                                                        "Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad")
            print(string)
            f.write(string + "\n")

            for i in range(nInput + nOutput):                
                cudart.cudaFree(bufferD[i])

#-------------------------------------------------------------------------------
def testDecoder():
    print("Test Decoder Part!")

    with open(decoderScoreFile, 'w') as f:

        if os.path.isfile(decoderPlanFile):
            with open(decoderPlanFile, 'rb') as decoderF:
                engine = trt.Runtime(logger).deserialize_cuda_engine(decoderF.read())
            if engine is None:
                print("Failed loading %s"%decoderPlanFile)
                return
            print("Succeeded loading %s"%decoderPlanFile)
        else:
            print("Failed finding %s"%decoderPlanFile)
            return

        nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
        nOutput = engine.num_bindings - nInput
        context = engine.create_execution_context()

        print(tableHead)  # for standard output

        for ioFile in sorted(glob(dataFilePath + "./decoder-*.npz")): 
            ioData = np.load(ioFile)
            encoder_out = ioData['encoder_out']
            encoder_out_lens = ioData['encoder_out_lens']
            hyps_pad_sos_eos = ioData['hyps_pad_sos_eos']
            hyps_lens_sos = ioData['hyps_lens_sos']
            ctc_score = ioData['ctc_score']
            batchSize, sequenceLength, _ = encoder_out.shape
            if batchSize > 16 or sequenceLength > 256:
                continue

            self_mask, cross_mask = gen_decoder_mask(hyps_lens_sos.reshape(-1).tolist(), 
                                            encoder_out_lens.tolist(),
                                            hyps_pad_sos_eos.shape[2]-1, encoder_out.shape[1])
            # print(hyps_lens_sos)
            # print(encoder_out_lens)
            # print(hyps_pad_sos_eos.shape, encoder_out.shape)
            # print(self_mask.shape, cross_mask.shape)
            context.set_binding_shape(0, encoder_out.shape)
            context.set_binding_shape(1, encoder_out_lens.shape)
            context.set_binding_shape(2, hyps_pad_sos_eos.shape)
            context.set_binding_shape(3, hyps_lens_sos.shape)
            context.set_binding_shape(4, ctc_score.shape)
            context.set_binding_shape(5, self_mask.shape)
            context.set_binding_shape(6, cross_mask.shape)
            #for i in range(nInput + nOutput):
            #    print("Input ->" if engine.binding_is_input(i) else "Output->", engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_dtype(i), engine.get_binding_name(i))
            #print("Finish all input binding: %s"%context.all_binding_shapes_specified)

            bufferH = []
            bufferH.append( encoder_out.astype(np.float32).reshape(-1) )
            bufferH.append( encoder_out_lens.astype(np.int32).reshape(-1) )
            bufferH.append( hyps_pad_sos_eos.astype(np.int32).reshape(-1) )
            bufferH.append( hyps_lens_sos.astype(np.int32).reshape(-1) )        
            bufferH.append( ctc_score.astype(np.float32).reshape(-1) )   
            bufferH.append( self_mask.astype(np.float32).reshape(-1) )   
            bufferH.append( cross_mask.astype(np.float32).reshape(-1) )

            for i in range(nInput, nInput + nOutput):                
                bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

            bufferD = []
            for i in range(nInput + nOutput):                
                bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            # warm up
            for i in range(10):
                context.execute_v2(bufferD)

            # test infernece time
            t0 = time_ns()
            for i in range(30):
                context.execute_v2(bufferD)
            t1 = time_ns()
            timePerInference = (t1-t0)/1000/1000/30

            indexDecoderOut = engine.get_binding_index('decoder_out')
            indexBestIndex = engine.get_binding_index('best_index')

            check0 = check(bufferH[indexDecoderOut],ioData['decoder_out'], True,)
            check1 = check(bufferH[indexBestIndex],ioData['best_index'], True)

            string = "%4d,%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e, %s"%(batchSize,
                                                                        sequenceLength,
                                                                        timePerInference,
                                                                        batchSize*sequenceLength/timePerInference*1000,
                                                                        check0[1],
                                                                        check0[2],
                                                                        check1[1],
                                                                        check1[2],
                                                                        "Good" if check0[1] < 4e-1 and check0[2] < 2e-4 and check1[2] < 1e-1 else "Bad")
            print(string)
            f.write(string + "\n")
            # if sequenceLength==256: exit(0)
            for i in range(nInput + nOutput):                
                cudart.cudaFree(bufferD[i])

if __name__ == "__main__":
    testEncoder()
    testDecoder()
