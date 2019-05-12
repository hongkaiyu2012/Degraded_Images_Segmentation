import os
import fcn
import math
import pytz
import tqdm
import torch
import utils
import shutil
import datetime
import scipy.misc
import numpy as np
import os.path as osp
from collections import OrderedDict
from torch.autograd import Variable


class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, test_loader, out, max_iter,
                 size_average=False, interval_validate=None):

        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone('America/New_York'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
        self.best_train_meanIoU = 0

        self.t_logger = utils.Logger(self.out, 'train')
        self.v_logger = utils.Logger(self.out, 'valid')
        self.ts_logger = utils.Logger(self.out, 'test')

    def test(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.test_loader.dataset.class_names)

        test_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Test iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            loss = utils.cross_entropy2d(score, target, size_average=self.size_average)
            loss_data = float(loss.data[0])
            if np.isnan(loss_data):
                raise ValueError('loss is nan while testing')
            test_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.uint8)
            lbl_true = target.data.cpu().numpy()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.test_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter_test_%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        test_loss /= len(self.test_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + [test_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # logging information for tensorboard
        info = OrderedDict({
            "loss": test_loss,
            "acc": metrics[0],
            "acc_cls": metrics[1],
            "meanIoU": metrics[2],
            "fwavacc": metrics[3],
            "bestIoU": self.best_mean_iu,
        })
        len(self.train_loader)
        # msg = "\t".join([key + ":" + "%.4f" % value for key, value in info.items()])
        partial_epoch = self.iteration / len(self.train_loader)
        for tag, value in info.items():
            self.ts_logger.scalar_summary(tag, value, partial_epoch)

        if training:
            self.model.train()

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            loss = utils.cross_entropy2d(score, target, size_average=self.size_average)
            loss_data = float(loss.data[0])
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :].astype(np.uint8)
            lbl_true = target.data.cpu().numpy()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter_val_%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        # logging information for tensorboard
        info = OrderedDict({
            "loss": val_loss,
            "acc": metrics[0],
            "acc_cls": metrics[1],
            "meanIoU": metrics[2],
            "fwavacc": metrics[3],
            "bestIoU": self.best_mean_iu,
        })
        len(self.train_loader)
        # msg = "\t".join([key + ":" + "%.4f" % value for key, value in info.items()])
        partial_epoch = self.iteration / len(self.train_loader)
        for tag, value in info.items():
            self.v_logger.scalar_summary(tag, value, partial_epoch)

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()
                self.test()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)
            weights = torch.from_numpy(self.train_loader.dataset.class_weights).float().cuda()
            ignore = self.train_loader.dataset.class_ignore
            loss = utils.cross_entropy2d(score, target, weight=weights, size_average=self.size_average, ignore=ignore)
            loss /= len(data)
            loss_data = float(loss.data[0])
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = utils.label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            # logging to tensorboard
            self.best_train_meanIoU = max(self.best_train_meanIoU, metrics[2])
            info = OrderedDict({
                "loss": loss.data[0],
                "acc": metrics[0],
                "acc_cls": metrics[1],
                "meanIoU": metrics[2],
                "fwavacc": metrics[3],
                "bestIoU": self.best_train_meanIoU,
            })
            partialEpoch = self.epoch + float(batch_idx) / len(self.train_loader)
            for tag, value in info.items():
                self.t_logger.scalar_summary(tag, value, partialEpoch)

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
