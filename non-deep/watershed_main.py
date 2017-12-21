from classifier import Classifier,Watershed
from data import baidu_VH,AsyncReader
from configs import *
from utils import *
from sklearn.externals import joblib
from log import log

def train():
    if args.dataset=='baidu_VH':
        dataset=baidu_VH(PROJECT_METAROOT)
    elif args.dataset=='summe':
        pass
        #dataset=
    else:
        raise ValueError('No such dataset')
    log.l.info(dataset.print_info())
    train_data=AsyncReader(dataset,root_path=BAIDU_VH_ROOT,mode='train',modality=args.modality)
    train_data.set_params({'limitedfiles':None,
                           'sample_rate':100,
                           'save_path':'tmp_results/train_{}_sampled.pkl'.format(args.modality)})
    X_train,Y_train=train_data.read_data(k=args.thread)

    val_data=AsyncReader(dataset,root_path=BAIDU_VH_ROOT,mode='val',modality=args.modality)
    val_data.set_params({'limitedfiles':None,
                           'sample_rate':1,
                           'save_path':'tmp_results/val_{}_sampled.pkl'.format(args.modality)})
    X_val,Y_val=val_data.read_data(k=args.thread)


    model=Classifier(model_name=args.model_name,if_grid_search=args.if_grid_search,model_kernel=args.model_kernel)
    if args.if_grid_search:
        model.set_grid_search_params(grid_search_params[args.model_name])
        X_train_grid_search,Y_train_grid_search=Sample_data(X_train,Y_train,args.grid_search_sample_rate)
        model.grid_search(X_train_grid_search,Y_train_grid_search)
    model.fit(X_train,Y_train)

    X_val_metric,Y_val_metric=Sample_data(X_val,Y_val,0.1)
    predict_val=model.predict(X_val_metric)
    metrics=get_metrics(predict_val,Y_val_metric,metrics=METRICS)
    # print metrics
    log.l.info('the metrics of {} is :{}'.format(METRICS,metrics))
    del X_train,Y_train#,X_train_grid_search,Y_train_grid_search,X_val_metric,Y_val_metric
    if args.create_curves:
    # for test set:
        val_curves_dic=dict()
        for k,v in val_data.data_dic.items():
            val_curves_dic[k]=model.predict(v)

        test_data=AsyncReader(dataset,root_path=BAIDU_VH_ROOT,mode='test',modality=args.modality)
        test_data.set_params({'limitedfiles':None,
                               'sample_rate':1,
                               'save_path':'tmp_results/test_{}_sampled.pkl'.format(args.modality)})
        _,_=test_data.read_data(k=args.thread)

        test_curves_dic=dict()
        for k,v in test_data.data_dic.items():
            test_curves_dic[k]=model.predict(v)
        return_info={'val':val_curves_dic,
                     'test':test_curves_dic}
        if args.save_curves:
            joblib.dump(return_info,'tmp_results/val_test_{}_curves.pkl'.format(args.modality))
        return return_info
    return None

def test(Infos):
    if Infos == None:
        if not os.path.exists('tmp_results/val_test_{}_curves.pkl'.format(args.modality)):
            exit()
        Infos=joblib.load('tmp_results/val_test_{}_curves.pkl'.format(args.modality))
        log.l.info('==========>  loaded the val_test_{}_curves.pkl from local....'.format(args.modality))
    if args.v: visualize(Infos['val'])

    watershed=Watershed(type_=args.type,num_classes=2)
    proposals_val=watershed.get_proposals(Infos['val'],alter=args.alter)
    proposals_test=watershed.get_proposals(Infos['test'],alter=args.alter)
    
    proposals_val=add_scores(proposals_val,Infos['val'])
    proposals_test=add_scores(proposals_test,Infos['test'])
    
    if not args.nms==0:
        proposals_val=temporal_nms(np.array(proposals_val),thresh=args.nms)
        proposals_test=temporal_nms(np.array(proposals_test),thresh=args.nms)
    if args.get_final_test:
        write_final_result_json(proposals_val,args.save_path.format('val'))
        write_final_result_json(proposals_test,args.save_path.format('test'))
        #get the evaluation result of validation set #
    

    #print done!

def main():
    global args,dataset
    args=get_args(mode='watershed')
    process=args.process
    Infos=None # represent the result of training process.
    if 'train' in process:
        Infos=train()
    if 'test' in process:
        test(Infos)

if __name__=='__main__':
    main()