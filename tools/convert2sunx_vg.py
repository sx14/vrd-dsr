import cPickle
import pickle
import scipy.io
import numpy as np

dataset = 'vg'
target = 'pre'

if dataset == 'vrd':
    from lib.label_hier.vrd.pre_hier import prenet
    from lib.label_hier.vrd.obj_hier import objnet
else:
    from lib.label_hier.vg.pre_hier import prenet
    from lib.label_hier.vg.obj_hier import objnet

with open('test_%s_%s.bin' % (target, dataset), 'rb') as f:
    pred = pickle.load(f)
pred_rlt_labels = pred['rlp_labels_ours']
pred_rlt_confs = pred['rlp_confs_ours']
pred_obj_boxes = pred['obj_bboxes_ours']
pred_sbj_boxes = pred['sub_bboxes_ours']


gt_path = '../data/vg/test.pkl'
with open(gt_path, 'rb') as f:
    gt = cPickle.load(f)

img_ids = []
for i in range(len(gt)):
    img_anno = gt[i]
    img_path = img_anno['img_path']
    img_id = img_path.split('/')[-1].split('.')[0]
    img_ids.append(img_id)

pred_roidb = {}

# object 1 base
# prdicate 0 base
raw_obj_labels = objnet.get_raw_labels()
raw_pre_labels = prenet.get_raw_labels()[1:]
for i in range(len(img_ids)):

    if pred_sbj_boxes[i] is None or \
            pred_sbj_boxes[i].shape[0] != pred_rlt_confs[i].shape[0]:
        continue

    img_id = img_ids[i]

    sbj_boxes = pred_sbj_boxes[i].astype(np.int)
    obj_boxes = pred_obj_boxes[i].astype(np.int)

    if sbj_boxes.shape[0] == 0:
        continue

    pre_boxes = np.concatenate((
        np.min(np.concatenate((sbj_boxes[:, 0:1], obj_boxes[:, 0:1]), axis=1), axis=1, keepdims=True),
        np.min(np.concatenate((sbj_boxes[:, 1:2], obj_boxes[:, 1:2]), axis=1), axis=1, keepdims=True),
        np.max(np.concatenate((sbj_boxes[:, 2:3], obj_boxes[:, 2:3]), axis=1), axis=1, keepdims=True),
        np.max(np.concatenate((sbj_boxes[:, 3:4], obj_boxes[:, 3:4]), axis=1), axis=1, keepdims=True)), axis=1)

    pred_sbj = pred_rlt_labels[i][:, 0:1]
    pred_pre = pred_rlt_labels[i][:, 1:2]
    pred_obj = pred_rlt_labels[i][:, 2:3]

    for j in range(pred_sbj.shape[0]):
        sbj = pred_sbj[j, 0]
        sbj_label = raw_obj_labels[int(sbj)]
        sbj_node = objnet.get_node_by_name(sbj_label)
        sbj_cls = sbj_node.index()
        pred_sbj[j, 0] = sbj_cls

        obj = pred_obj[j, 0]
        obj_label = raw_obj_labels[int(obj)]
        obj_node = objnet.get_node_by_name(obj_label)
        obj_cls = obj_node.index()
        pred_obj[j, 0] = obj_cls

        pre = pred_pre[j, 0]
        pre_label = raw_pre_labels[int(pre)]
        pre_node = prenet.get_node_by_name(pre_label)
        pre_cls = pre_node.index()
        pred_pre[j, 0] = pre_cls


    pred_confs = pred_rlt_confs[i]
    pred_confs = pred_confs[:, np.newaxis]

    pred_rois = np.concatenate((pre_boxes, pred_pre,
                                sbj_boxes, pred_sbj,
                                obj_boxes, pred_obj,
                                pred_confs), axis=1)
    pred_roidb[img_id] = pred_rois

save_path = '%s_box_label_%s_dsr.bin' % (target, dataset)
with open(save_path, 'wb') as f:
    pickle.dump(pred_roidb, f)