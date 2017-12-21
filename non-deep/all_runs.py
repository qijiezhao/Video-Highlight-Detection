import os,sys
from itertools import combinations
order='python watershed_main.py --process test --dataset baidu_VH --thread 100 --model_name xgb --modality fusion --type smooth --alter {} --save_path tmp_results/final_{}.json'

raw_number=[4,12,24,40,80,100,150,200]
alters=[]
for i in range(1,len(raw_number)+1):
    alters.extend(list(combinations(raw_number,i)))

print alters

for i,alter in enumerate(alters):
    print i+1
    command=order.format(' '.join([str(_) for _ in alter]),i+1)
    os.system(command)

