import sys
import numpy as np
from pathlib import Path


path = Path(sys.argv[1])
backend = sys.argv[2]
method = sys.argv[3]

# path = Path('/workspace/wenet_trt8/log/npys')
# backend = 'engine1'
# method = 'attention_rescoring'

array = np.load(path / f'{backend}_{method}.npy')

nums = array[:,0].shape[0]
encoder_times = array[:,1]
decoder_times = array[:,2]

encoder_mean = encoder_times.sum()/nums
decoder_mean = decoder_times.sum()/nums

print(f'Backend: {backend}\nMethod: {method}\nMean(ms): \nencoder ——— {encoder_mean}\ndecoder ——— {decoder_mean}')



