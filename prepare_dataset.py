import json
import tarfile
from pathlib import Path
import sys

path = Path(sys.argv[1])
test_tar = path / 'datasets' /'test.tar'
data_list = path / 'datasets' / 'test_data.list'
data_dir = test_tar.parent / test_tar.stem
data_dict = {json.loads(i)['key']:json.loads(i) for i in data_list.read_text().splitlines()}

if not data_dir.exists():
    file = tarfile.open(test_tar,mode = "r")
    file.extractall(path=path / 'datasets')

with open(path / 'datasets' / 'data.list','w+') as f:
    for i in data_dir.iterdir():
        if i.is_dir():
            for j in i.iterdir():
                data_dict[j.stem]['wav'] = str((data_dir/data_dict[j.stem]['wav']).absolute())
                f.write(json.dumps(data_dict[j.stem],ensure_ascii=False)+"\n")
