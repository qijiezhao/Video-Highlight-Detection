import os,sys
import numpy as np
import json
from multiprocessing import Pool
from sklearn.externals import joblib
from log import log

def read_data_instance(file_name,file_gt,modality,params):
    if modality in ['audio','image']:
        file_path=os.path.join(params['root_data_{}'.format(modality)],'{}.pkl'.format(file_name))
        data_raw=joblib.load(file_path)
    else: # modality=='fusion':
        file_path_audio=os.path.join(params['root_data_{}'.format('audio')],'{}.pkl'.format(file_name))
        file_path_image=os.path.join(params['root_data_{}'.format('image')],'{}.pkl'.format(file_name))
        data_audio=joblib.load(file_path_audio)
        data_image=joblib.load(file_path_image)
        data_raw=np.concatenate([data_audio,data_image],axis=1)
    len_data=len(data_raw)
    if not file_gt==None:
        GT=np.array([_['segment'] for _ in file_gt['annotations']]).astype(np.int)
        label_=np.zeros(len_data)
        for gt in GT: label_[gt[0]:gt[1]]=1
    else:
        label_=np.zeros(len_data)
    sample_inds=np.array(range(0,len_data,params['sample_rate']))
    sys.stdout.flush()

    return {'data':data_raw[sample_inds],'label':label_[sample_inds],'file_name':file_name}

class AsyncReader(object):
    def __init__(self,dataset,root_path,mode,modality):
        self.dataset=dataset
        self.root_path=root_path
        self.mode=mode
        self.read_data_instance_f=read_data_instance
        self.modality=modality

    def set_params(self,params):
        self.limitedfiles=0
        if not params['limitedfiles']==None:
            self.limitedfiles=params['limitedfiles'] # the most data to read
        self.sample_rate=params['sample_rate']
        self.save_path=params['save_path']

        pass


    def read_data(self,k=8):

        data_list=getattr(self.dataset,'{}_list'.format(self.mode))
        if not self.limitedfiles==0: data_list=data_list[:self.limitedfiles]
        if not self.mode=='test':
            data_GT=getattr(self.dataset,'{}_GT'.format(self.mode))

        if os.path.exists(self.save_path):
            log.l.info('=====> dataset of {} already exists, directly loading it from local...'.format(self.mode))
            data_label=joblib.load(self.save_path)
            self.data_dic=data_label['data_dic']
            self.label_dic=data_label['label_dic']
            self.data=np.concatenate([self.data_dic[_] for _ in data_list])
            self.label=np.concatenate([self.label_dic[_] for _ in data_list])
            self.len_list=np.array([len(self.label_dic[_]) for _ in data_list])
            return self.data,self.label

        self.root_data=os.path.join(self.root_path,'{}_{}'.format(self.mode,self.modality))



        self.data_dic=dict()
        self.label_dic=dict()

        params={'root_data_audio':os.path.join(self.root_path,'{}_{}'.format(self.mode,'audio')),
                'root_data_image':os.path.join(self.root_path,'{}_{}'.format(self.mode,'image')),
                'sample_rate':self.sample_rate}  # this param is for read_data_instance
        pool = Pool(k)
        jobs=list()
        log.l.info('=====> loading data {} from dataroot with {} multi-threads'.format(self.mode,k))
        for i,file_name in enumerate(data_list):
            gt_tmp=data_GT[file_name] if not self.mode=='test' else None
            jobs.append(pool.apply_async(self.read_data_instance_f,args=(file_name,gt_tmp,self.modality,params),callback=self.callback))

        pool.close()
        pool.join()
        
        self.data=np.concatenate([self.data_dic[_] for _ in data_list])
        self.label=np.concatenate([self.label_dic[_] for _ in data_list])
        self.len_list=np.array([len(self.label_dic[_]) for _ in data_list])

        log.l.info('=====> read data: {} done, saving it now...'.format(self.mode))
        joblib.dump({'data_dic':self.data_dic,
                     'label_dic':self.label_dic},self.save_path)

        return self.data,self.label


    def callback(self,rst):
        sys.stdout.flush()
        self.data_dic[rst['file_name']]=rst['data']
        self.label_dic[rst['file_name']]=rst['label']



class baidu_VH(object):
    def __init__(self,mete_root):  # mete_root is the root path of mete file
        self.mete_root=mete_root
        self.mete_file=os.path.join(self.mete_root,'mete.json')
        if not os.path.exists(self.mete_file):
            exit()
        self.parse_json()

    def parse_json(self):
        data=json.load(open(self.mete_file))
        self.version=data['version']
        self.database=data['database']
        del data

        self.train_list=[k for k,v in self.database.items() if v['subset']=='training']
        self.val_list=[k for k,v in self.database.items() if v['subset']=='validation']
        self.test_list=[k for k,v in self.database.items() if v['subset']=='testing']

        self.trainval_list=self.train_list+self.val_list
        self.all_list=self.trainval_list+self.test_list

        self.train_GT={k:v for k,v in self.database.items() if k in self.train_list}
        self.val_GT={k:v for k,v in self.database.items() if k in self.val_list}

        self.trainval_GT={k:v for k,v in self.database.items() if k in self.trainval_list}


    def print_info(self):
        len_train=len(self.trainval_list)
        len_val=len(self.val_list)
        len_test=len(self.test_list)

        n_train_instances=sum([len(_['annotations']) for _ in self.train_GT.values()])
        n_val_instances=sum([len(_['annotations']) for _ in self.val_GT.values()])

        return '''
        ===========================     Have loaded the metafile     =========================
        train and val infos :
            n_train: {}
            n_val: {}
            n_test: {}
            n_train_instance: {}({})
            n_val_instance: {}/({})
        '''.format(len_train,len_val,len_test,n_train_instances,\
                   n_train_instances/float(len_train),n_val_instances,\
                   n_val_instances/float(len_val))


