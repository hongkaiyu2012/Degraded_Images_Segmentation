import fcn
import tqdm
import scipy.misc
from torch.autograd import Variable

from utils import *


class Tester(object):

    def __init__(self, cuda, model, test_data, test_loader, out):
        self.cuda = cuda
        self.model = model
        self.test_data = test_data
        self.test_loader = test_loader
        self.out = out
        self.timestamp_start = datetime.datetime.now(pytz.timezone('America/New_York'))

        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'test/loss',
            'test/acc',
            'test/acc_cls',
            'test/mean_iu',
            'test/fwavacc',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

    def test(self):
        self.model.eval()

        out_pred = osp.join(self.out, 'pred')
        out_pred_color = osp.join(self.out, 'pred_color')
        if not osp.exists(out_pred):
            os.makedirs(out_pred)
        if not osp.exists(out_pred_color):
            os.makedirs(out_pred_color)

        n_class = len(self.test_loader.dataset.class_names)

        test_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        idx = 0
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.test_loader), total=len(self.test_loader),
                                                   ncols=80, leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=False)
            loss_data = float(loss.data[0])
            if np.isnan(loss_data):
                raise ValueError('loss is nan while testing')
            test_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.test_loader.dataset.untransform(img, lt)
                lpc = self.test_data.label_to_color_image(lp)
                label_trues.append(lt)
                label_preds.append(lp)

                name = self.test_data.files['test'][batch_idx]['lbl'].split('/')[-1]
                out_file_lp = osp.join(out_pred, name)
                out_file_lpc = osp.join(out_pred_color, name)
                idx += 1

                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
                lp = scipy.misc.toimage(lp, cmin=0, cmax=255)
                scipy.misc.imsave(out_file_lp, lp)
                scipy.misc.imsave(out_file_lpc, lpc)
        metrics = label_accuracy_score(label_trues, label_preds, n_class)

        out_file = osp.join(self.out, 'visualization.jpg')
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        test_loss /= len(self.test_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            log = [test_loss] + list(metrics)
            log = map(str, log)
            f.write(','.join(log) + '\n')
