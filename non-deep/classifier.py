import os,sys

from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn import preprocessing
from configs import *
from log import log

supported_models=['xgb','svm','lr']

class Classifier(object):
    def __init__(self,model_name,if_grid_search,model_kernel='linear'):
        self.model_name=model_name
        self.if_grid_search=if_grid_search
        self.model_kernel=model_kernel
        if not self.model_name in supported_models:
            exit()
        log.l.info('=========================   building the model {} ======================'.format(self.model_name))
        self._build_model(self.model_name,params=None)

    def set_grid_search_params(self,params):
        self.params=params
        log.l.info('setted the grid search params: {}'.format(
            '\n'.join(['====>  {}:{}'.format(k,v) for k,v in params.items()])))

    def _build_model(self,model_name,params=None):
        if params==None:
            if model_name=='xgb':
                self.model=XGBClassifier(n_estimators=100,learning_rate=0.02)
            elif model_name=='svm':
                kernel_function=chi2_kernel if not (self.model_kernel=='linear' or self.model_kernel=='rbf') else self.model_kernel
                self.model=SVC(C=1,kernel=kernel_function,gamma=1,probability=True)
            elif model_name=='lr':
                self.model=LR(C=1,penalty='l1',tol=1e-6)
        else:
            if model_name=='xgb':
                self.model=XGBClassifier(n_estimators=1000,learning_rate=0.02,**params)
            elif model_name=='svm':
                self.model=SVC(C=1,kernel=kernel_function,gamma=1,probability=True)
            elif model_name=='lr':
                self.model=LR(C=1,penalty='l1',tol=1e-6)

        log.l.info('=======> built the model {} done'.format(self.model_name))

    def grid_search(self,x,y,verbose=1,scoring='average_precision',cv=5,n_jobs=10):
        if not self.if_grid_search:
            log.l.info('Not allowed grid search, please check')
            exit()

        model_grid=GridSearchCV(self.model,self.params,
                                n_jobs=n_jobs,verbose=verbose,cv=cv,scoring=scoring)
        log.l.info('\n=======> Doing grid search...')
        model_grid.fit(x,y)
        model_best_score=self.model_grid.best_score_
        self.model_best_params=self.model_grid.best_params_

    def fit(self,x,y):
        log.l.info('=======> fitting...')
        if self.if_grid_search:
            self.model=self._build_model(**self.model_best_params)
        if self.model_name=='svm' and self.model_kernel=='x2':
            x-=MIN_FEATURE
            x=preprocessing.normalize(x,norm='l1')
        self.model.fit(x,y)

    def predict(self,x):
        log.l.info('=======> predicting...')
        if self.model_name=='svm' and self.model_kernel=='x2':
            x-=MIN_FEATURE
            x=preprocessing.normalize(x,norm='l1')
        return self.model.predict_proba(x)



class Watershed(object):
    def __init__(self,type_='clips',num_classes=2):
        self.type=type_
        self.num_classes=num_classes

    def get_proposals(self,video_score_dic,alter=None):
        if alter==None:return 0
        if self.type=='clips':
            return self.get_proposals_clips(video_score_dic,alter)
        elif self.type=='smooth':
            return self.get_proposals_smooth(video_score_dic,alter)

    def get_proposals_smooth(self,video_score_dic,alter):
        proposals=list()
        for smooth_term in alter:
            '''
            clips=[0.95,0.5,0.25,0.15,0.05,0.01]
            '''
            for i,(video,score) in enumerate(video_score_dic.items()):
                tmp_score=score.copy()
                tmp_score=self.smooth(tmp_score,smooth_term)
                tmp_score[tmp_score<1]=0
                tmp_score[tmp_score>1]=1
                start_end_list=self.get_s_e(tmp_score,video)
                proposals.extend(start_end_list)

            print('smooth term {} has done!'.format(smooth_term))
        return proposals
    def get_proposals_clips(self,video_score_dic,alter):
        proposals=list()
        for clip in alter:
            '''
            clips=[0.95,0.5,0.25,0.15,0.05,0.01]
            '''
            for i,video,score in enumerate(video_score_dic.items()):
                tmp_score=score.copy()
                tmp_score[tmp_score<clip]=0
                tmp_score[tmp_score>clip]=1
                start_end_list=self.get_s_e(tmp_score,video)
                proposals.extend(start_end_list)

            print('clip {} has done!'.format(clip))
        return proposals

    def smooth(self,old_score,terms):
        smoothing_vec=np.ones(terms)
        sum_smooth_vec=terms
        new_scores=np.zeros_like(old_score)
        old_score=np.concatenate([old_score[0].reshape(1,-1).repeat(len(smoothing_vec)/2,0),\
                                 old_score,\
                                 old_score[-1].reshape(1,-1).repeat(len(smoothing_vec)/2-1,0)])  # padding with repeat
        for i in range(len(new_scores)):
            new_scores[i]=np.dot(old_score[i:i+len(smoothing_vec)].T,smoothing_vec)/sum_smooth_vec
        return new_scores

    def get_s_e(self,score_ins,video):
        s_e_list=list()
        for i in range(1,self.num_classes):
            s,e=0,0;lock=0
            score_item=score_ins[:,i] # each class
            for j in range(len(score_item)):
                if lock==0 and score_item[j]!=0:
                    s=j
                    lock=1
                if lock==1 and score_item[j]==0:
                    e=j
                    s_e_list.append([video,s,e,i])
                    lock=0
        return s_e_list
        #return self.post(s_e_list,score_ins,video) # to ensemble by curves


    def post(self,s_e_list,score_ins,video):
        posted_s_e_list=s_e_list
        for ii in range(1,self.num_classes):
            tmp_s_e_lists=[_ for _ in s_e_list if _[3]==ii]
            s_s=[_[1] for _ in tmp_s_e_lists]
            e_s=[_[2] for _ in tmp_s_e_lists]

            for i,s_ in enumerate(s_s):
                for j,e_ in enumerate(e_s):
                    if i<j and s_<e_:
                        if sum(score_ins[s_:e_,ii])/float((e_-s_))>0.9:
                            posted_s_e_list.append([video,s_,e_,ii])
        return posted_s_e_list



#class NeuralNetwork(object):
