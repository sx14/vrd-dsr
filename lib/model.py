# coding: utf-8

import time
import cPickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable


from lib.utils import AverageMeter
from lib.evaluation import eval_reall_at_N, eval_obj_img
from lib.data_layers.vrd_data_layer import VrdDataLayer

def train_net(train_data_layer, net, epoch, args):
    net.train()
    losses = AverageMeter()
    time1 = time.time()
    epoch_num = train_data_layer._num_instance/train_data_layer._batch_size
    for step in range(epoch_num):
        image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, rel_labels, rel_so_prior = train_data_layer.forward()
        target = Variable(torch.from_numpy(rel_labels).type(torch.LongTensor)).cuda()
        rel_so_prior = -0.5*(rel_so_prior+1.0/args.num_relations)
        rel_so_prior = Variable(torch.from_numpy(rel_so_prior).type(torch.FloatTensor)).cuda()
        # forward
        args.optimizer.zero_grad()
        obj_score, rel_score = net(image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, args)
        loss = args.criterion((rel_so_prior+rel_score).view(1, -1), target)
        losses.update(loss.data[0])
        loss.backward()
        args.optimizer.step()
        if step % args.print_freq == 0:
            time2 = time.time()
            print "TRAIN:%d, Total LOSS:%f, Time:%s" % (step, losses.avg, time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
            time1 = time.time()
            losses.reset()


def test_pre_net(net, args):
    net.eval()
    time1 = time.time()
    res = {}
    rlp_labels_ours  = []
    tuple_confs_cell = []
    sub_bboxes_cell  = []
    obj_bboxes_cell  = []
    test_data_layer = VrdDataLayer(args.ds_name, 'test', model_type = args.model_type)
    # for step in range(1000):
    for step in range(test_data_layer._num_instance):
        test_data = test_data_layer.forward()
        if(test_data is None):
            rlp_labels_ours.append(None)
            tuple_confs_cell.append(None)
            sub_bboxes_cell.append(None)
            obj_bboxes_cell.append(None)
            continue

        # img(resized), detection(resized), union box(resized)
        # 空间特征，detection classes, sbj-det-inds, obj-det,inds, detection(org)
        image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, ori_bboxes = test_data

        # 用于保存三元组
        rlp_labels_im  = np.zeros((100, 3), dtype = np.float)
        # 用于保存三元组conf
        tuple_confs_im = []

        sub_bboxes_im  = np.zeros((100, 4), dtype = np.float)
        obj_bboxes_im  = np.zeros((100, 4), dtype = np.float)

        # 预测
        # Attention: obj_scores没有使用
        obj_score, rel_score = net(image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, args)
        rel_prob = rel_score.data.cpu().numpy()

        # rel_res = []
        # for tuple_idx in range(rel_prob.shape[0]):
        #     probs = rel_prob[tuple_idx]
        #     rel = np.argmax(probs)
        #     rel_res.append([tuple_idx, rel])
        # rel_res = np.array(rel_res)

        # rlp_labels_im = rlp_labels_im[:rel_res.shape[0], :]
        # sub_bboxes_im = sub_bboxes_im[:rel_res.shape[0], :]
        # obj_bboxes_im = obj_bboxes_im[:rel_res.shape[0], :]


        aaa = np.argsort(-rel_prob.ravel())
        bbb = np.unravel_index(aaa, rel_prob.shape)
        ccc = np.dstack(bbb)
        ddd = ccc[0]
        rel_res = ddd[:100]


        # 这一句是填充50/100的
        # rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:100]
        for ii in range(rel_res.shape[0]):
            rel = rel_res[ii, 1]
            tuple_idx = rel_res[ii, 0]
            conf = rel_prob[tuple_idx, rel]
            sub_bboxes_im[ii] = ori_bboxes[ix1[tuple_idx]]
            obj_bboxes_im[ii] = ori_bboxes[ix2[tuple_idx]]
            rlp_labels_im[ii] = [classes[ix1[tuple_idx]], rel, classes[ix2[tuple_idx]]]
            tuple_confs_im.append(conf)
        if(args.ds_name =='vrd'):
            rlp_labels_im += 1
        tuple_confs_im = np.array(tuple_confs_im)
        rlp_labels_ours.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)
    res['rlp_labels_ours'] = rlp_labels_ours
    res['rlp_confs_ours'] = tuple_confs_cell
    res['sub_bboxes_ours'] = sub_bboxes_cell
    res['obj_bboxes_ours'] = obj_bboxes_cell
    rec_50  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = False)
    rec_50_zs  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = True)
    rec_100 = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = False)
    rec_100_zs = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = True)
    print 'CLS TEST r50:%f, r50_zs:%f, r100:%f, r100_zs:%f'% (rec_50, rec_50_zs, rec_100, rec_100_zs)
    time2 = time.time()
    print "TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
    return rec_50, rec_50_zs, rec_100, rec_100_zs


def test_rel_net(net, args):
    net.eval()
    time1 = time.time()
    pos_num = 0.0
    loc_num = 0.0
    gt_num = 0.0

    # 加载测试集GT
    with open('../data/%s/test.pkl'%args.ds_name, 'rb') as fid:
        anno = cPickle.load(fid)

    # VRD格式输出
    res = {}
    rlp_labels_ours  = []
    tuple_confs_cell = []
    sub_bboxes_cell  = []
    obj_bboxes_cell  = []
    test_data_layer = VrdDataLayer(args.ds_name, 'test', model_type = args.model_type, proposals_path = args.proposal)
    predict = []

    N_rlt_pred = 0

    # for step in range(1000):
    for step in range(test_data_layer._num_instance):
        test_data = test_data_layer.forward()
        if(test_data is None):
            rlp_labels_ours.append(None)
            tuple_confs_cell.append(None)
            sub_bboxes_cell.append(None)
            obj_bboxes_cell.append(None)
            predict.append(None)
            continue

        # img(resized), det box(resized), union box(resized)
        # 空间特征，detection classes, sbj-det-inds, obj-det,inds
        # det box(org), det confs, 先验
        image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, ori_bboxes, pred_confs, rel_so_prior = test_data
        N_rlt_pred += ix1.shape[0]

        # 测试
        # 物体检测得分，predicate得分
        obj_score, rel_score = net(image_blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, args)

        # 物体检测结果和得分，这里没有用
        _, obj_pred = obj_score[:, 1::].data.topk(1, 1, True, True)

        # 物体检测得分归一化
        obj_score = F.softmax(obj_score)[:, 1::].data.cpu().numpy()

        # 加载当前图像的object GT
        anno_img = anno[step]
        gt_boxes = anno_img['boxes'].astype(np.float32)
        gt_cls = np.array(anno_img['classes']).astype(np.float32)

        # eval物体检测部分
        # pos是类别和box都正确，loc是仅box正确
        pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, ori_bboxes, obj_pred.cpu().numpy(), gt_thr=0.5)
        # pos_num_img, loc_num_img = eval_obj_img(gt_boxes, gt_cls, ori_bboxes, classes, gt_thr=0.5)

        gt_num += gt_boxes.shape[0]
        pos_num += pos_num_img
        loc_num += loc_num_img
        rel_prob = rel_score.data.cpu().numpy()

        # predicate得分，加上先验得分
        # num_relations是predicate类别数
        # TODO:先验得分怎么算
        rel_prob += np.log(0.5*(rel_so_prior+1.0/args.num_relations))

        # 预测relationship的三元组，n*(n-1)*70组
        rlp_labels_im  = np.zeros((rel_prob.shape[0], 3), dtype = np.float)
        tuple_confs_im = []
        sub_bboxes_im  = np.zeros((rel_prob.shape[0], 4), dtype = np.float)
        obj_bboxes_im  = np.zeros((rel_prob.shape[0], 4), dtype = np.float)
        n_idx = 0

        for tuple_idx in range(rel_prob.shape[0]):
            sub = classes[ix1[tuple_idx]]
            obj = classes[ix2[tuple_idx]]

            rel_p = rel_prob[tuple_idx]
            rel = np.argmax(rel_p)

            if (pred_confs.ndim == 1):
                conf = np.log(pred_confs[ix1[tuple_idx]]) + np.log(pred_confs[ix2[tuple_idx]]) + rel_prob[tuple_idx, rel]
            else:
                conf = np.log(pred_confs[ix1[tuple_idx], 0]) + np.log(pred_confs[ix2[tuple_idx], 0]) + rel_prob[tuple_idx, rel]
            sub_bboxes_im[n_idx] = ori_bboxes[ix1[tuple_idx]]
            obj_bboxes_im[n_idx] = ori_bboxes[ix2[tuple_idx]]
            rlp_labels_im[n_idx] = [sub, rel, obj]
            tuple_confs_im.append(conf)
            n_idx += 1

        # rlp_labels_im = np.zeros((rel_prob.shape[0] * rel_prob.shape[1], 3), dtype=np.float)
        # tuple_confs_im = []
        # sub_bboxes_im = np.zeros((rel_prob.shape[0] * rel_prob.shape[1], 4), dtype=np.float)
        # obj_bboxes_im = np.zeros((rel_prob.shape[0] * rel_prob.shape[1], 4), dtype=np.float)
        # n_idx = 0
        #
        # for tuple_idx in range(rel_prob.shape[0]):
        #     sub = classes[ix1[tuple_idx]]
        #     obj = classes[ix2[tuple_idx]]
        #     for rel in range(rel_prob.shape[1]):
        #         if (args.use_obj_prior):
        #             # 使用物体得分
        #             # relationship 得分为sbj,obj,pre得分之和，即均值
        #             if (pred_confs.ndim == 1):
        #                 conf = np.log(pred_confs[ix1[tuple_idx]]) + np.log(pred_confs[ix2[tuple_idx]]) + rel_prob[
        #                     tuple_idx, rel]
        #             else:
        #                 conf = np.log(pred_confs[ix1[tuple_idx], 0]) + np.log(pred_confs[ix2[tuple_idx], 0]) + rel_prob[
        #                     tuple_idx, rel]
        #         else:
        #             # 不使用物体得分
        #             # relationship得分就是predicate得分
        #             conf = rel_prob[tuple_idx, rel]
        #         sub_bboxes_im[n_idx] = ori_bboxes[ix1[tuple_idx]]
        #         obj_bboxes_im[n_idx] = ori_bboxes[ix2[tuple_idx]]
        #         rlp_labels_im[n_idx] = [sub, rel, obj]
        #         tuple_confs_im.append(conf)
        #         n_idx += 1


        if(args.ds_name =='vrd'):
            # class 1 based
            rlp_labels_im += 1
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:100]
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        sub_bboxes_im  = sub_bboxes_im[idx_order,:]
        obj_bboxes_im  = obj_bboxes_im[idx_order,:]
        rlp_labels_ours.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)
    res['rlp_labels_ours'] = rlp_labels_ours
    res['rlp_confs_ours'] = tuple_confs_cell
    res['sub_bboxes_ours'] = sub_bboxes_cell
    res['obj_bboxes_ours'] = obj_bboxes_cell
    rec_50  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = False)
    rec_50_zs  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = True)
    rec_100 = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = False)
    rec_100_zs = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = True)
    print 'CLS OBJ TEST POS:%f, LOC:%f, GT:%f, Precision:%f, Recall:%f'% (pos_num, loc_num, gt_num, pos_num/(pos_num+loc_num), pos_num/gt_num)
    print 'CLS REL TEST r50:%f, r50_zs:%f, r100:%f, r100_zs:%f'% (rec_50, rec_50_zs, rec_100, rec_100_zs)
    time2 = time.time()
    print "TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
    print('pred rlt num: %d' % N_rlt_pred)
    return rec_50, rec_50_zs, rec_100, rec_100_zs
