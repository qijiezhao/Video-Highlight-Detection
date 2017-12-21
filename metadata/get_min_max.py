import os,sys
import json
file='mete.json'

data=json.load(open(file))['database']
min_=1000
max_=0
for file_name,gts in data.items():
    gts_ins=gts['annotations']
    for gt_ins in gts_ins:
        gap=gt_ins['segment'][1]-gt_ins['segment'][0]
        if gap>max_:max_=gap
        if gap<min_:min_=gap

print max_,min_
