import json
import tarfile
from pathlib import Path


test_tar = Path('./test.tar')
data_list = Path('./test_data.list')
data_dir = test_tar.parent / test_tar.stem
data_dict = {json.loads(i)['key']:json.loads(i) for i in data_list.read_text().splitlines()}

# print(datalist['BAC009S0765W0207.wav'])
# exit()


if not data_dir.exists():
    file = tarfile.open(test_tar,mode = "r")
    file.extractall(path='./')


with open('./data.list','w+') as f:
    for i in data_dir.iterdir():
        if i.is_dir():
            for j in i.iterdir():
                data_dict[j.stem]['wav'] = str((data_dir/data_dict[j.stem]['wav']).absolute())
                f.write(json.dumps(data_dict[j.stem],ensure_ascii=False)+"\n")
