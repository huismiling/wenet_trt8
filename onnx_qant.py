import os
import random
import onnx
import onnxruntime
import numpy as np
import json
from onnxruntime.quantization import CalibrationDataReader
import math

class onnxDataReader(CalibrationDataReader):
    def __init__(self,
                 npData,
                 batch_size,
                 ):
        self.npData = npData
        self.batch_size = batch_size
        self.enum_data_dicts = iter([])
        self.total_data_num = 10
        self.current_idx = 0

    def get_next(self):
        iter_data = next(self.enum_data_dicts, None)
        if iter_data:
            self.start_of_new_stride= False
            self.current_idx += 1
            return iter_data
        
        if self.total_data_num < self.current_idx:
            return None

        self.current_idx += 1

        self.enum_data_dicts = None
        data = []
        for itd in self.npData:
            dkeys = list(itd.keys())
            num_batch = math.ceil(len(itd[dkeys[0]])/self.batch_size)
            for itn in range(num_batch):
                ddict = {}
                for itk in dkeys:
                    ddict[itk] = itd[itk][itn*self.batch_size : (itn+1)*self.batch_size]
                data.append(ddict)
        
        random.shuffle(data)
        data = data[:self.total_data_num]

        self.enum_data_dicts = iter(self.npData)
        self.start_of_new_stride = True
        return next(self.enum_data_dicts, None)



