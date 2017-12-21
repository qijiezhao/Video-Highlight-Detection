from __future__ import print_function
import os,sys
import json
from data import baidu_VH
from configs import *
from utils import iou
file_root='../metadata/final_super_ultimate_{}.json'
target_root='../metadata/{}_list.txt'
mini_length=30
dataset=baidu_VH(PROJECT_ROOT)

for item in ['train','val','test']:
    proposal_file=file_root.format(item)
    target_file=target_root.format(item)
    name_list=getattr(dataset,'{}_list'.format(item))

    if not item=='test':
        GTs=getattr(dataset,'{}_GT'.format(item))

    proposal_dic=proposal_file['results']

    contents=list()

    for name in name_list:
        proposals=proposal_dic[name]
        proposals=[_['segment'] for _ in proposals] if _['segment'][1]-_['segment'][0]>mini_length]
        IoUs=[1]*len(proposals)
        if not item=='test':
            GT_ins=GTs[name]['annotations']
            GT_ins=[_['segment'] for _ in GT_ins]
            IoUs=wrapper_segment_iou(GT_ins,proposals)
            IoUs=np.max(IoUs,1)
        for i,iou in enumerate(IoUs):
            if iou>=0.5:
                contents.append('{} {} {} 1'.format(name,proposals[i][0],proposals[i][1]))
            else:
                contents.append('{} {} {} 0'.format(name,proposals[i][0],proposals[i][1]))
        print('file {} has done, posis:{} negas:{}'.format(name,len(contents_positive),len(contents_negative)))


    with open(target_file,'w')as fw:fw.write('\n'.join(contents))

print('generated train,val,test files done!')