import os,sys
import numpy as np
import random,json
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def Sample_data(X,Y,sample_rate):
    len_data=len(X)
    n_sample=int(len_data*sample_rate)
    sample_inds=np.array(random.sample(range(len_data),n_sample))
    return X[sample_inds],Y[sample_inds]

def add_scores(proposals,infos_dic):
    new_proposals=list()
    for proposal in proposals:
        video,s,e,i=tuple(proposal)
        score_tmp=sum(infos_dic[video][s:e,i])/float(e-s)
        new_proposals.append([video,s,e,i,score_tmp])
    return new_proposals

def write_final_result_txt(proposals,path):
    out_put=list()
    for proposal in proposals:
        video,s,e,i,c=proposal[0],proposal[1],proposal[2],proposal[3],proposal[4]
        out_='{} {} {} {} {}'.format(video,s,e,i,c)
        out_put.append(out_)
    with open('tmp_results/final_result.txt','w') as fw:fw.write('\n'.join(out_put))

def write_final_result_json(proposals,path):
    out_put=dict()
    out_put['version']='VERSION 1.0'
    out_put['results']=dict()
    for proposal in proposals:
        video,s,e,i,c=tuple(proposal)
        if e-s>30:
            if not video in out_put['results'].keys():
                out_put['results'][video]=[{'score':c,'segment':[s,e]}]
            else:
                out_put['results'][video].append({'score':c,'segment':[s,e]})
    with open(path,'w')as json_file:
        json.dump(out_put,json_file)

def get_metrics(predicted,gt,metrics='average_precision'):
    if metrics=='average_precision':
        return average_precision_score(gt,predicted[:,1])
def temporal_nms(bboxes, thresh):
    """
    One-dimensional non-maximal suppression
    :param bboxes: [[st, ed, score, ...], ...]
    :param thresh:
    :return:
    """
    t1 = np.array(bboxes[:, 1],dtype=np.int) # s
    t2 = np.array(bboxes[:, 2],dtype=np.int) # e 
    scores = np.array(bboxes[:, 4],dtype=np.float) # score

    durations = t2 - t1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        tt1 = np.maximum(t1[i], t1[order[1:]])
        tt2 = np.minimum(t2[i], t2[order[1:]])
        intersection = tt2 - tt1
        IoU = intersection / (durations[i] + durations[order[1:]] - intersection).astype(float)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return bboxes[keep, :]

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in xrange(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    return tiou

def visualize(scores):
    for k,v in scores.items():
        name=k
        value=v
        plt.plot(value)
        plt.title(name)
        plt.show()



#========================================= For deep models' ====================================#

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.recall=0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def add_rec(self, val):
        if isinstance(val,int):
            pass
        else:
            self.sum+=val[0]
            self.count+=val[1]
            self.recall=self.count/float(self.sum)

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def recall(output,target, conts=(1,2,3)):
    batch_size=target.size(0)
    max_inds=np.argmax(output.cpu().numpy(),1)
    res=[]
    target=target.cpu()
    for item in conts:
        inds=target==item
        if sum(inds)==0:res.append(-1)
        else:
            rt=sum(max_inds[inds.numpy().astype(bool)]==item)
            res.append([sum(inds),rt])
    return tuple(res)