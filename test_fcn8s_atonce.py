#!/usr/bin/env python

import datasets
import argparse
from utils import *
from network_tester import Tester


def main():
    # 0. input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Checkpoint path')
    parser.add_argument('-g', '--gpu', type=int, help='GPU device to use', default=0)
    parser.add_argument('-d', '--dataset', help='VOC, CamVid, SUNRGBD, Cityscapes, Custom', default='CamVid')
    parser.add_argument('-dr', '--datasetroot', help='dataset root pth', default='/home/hongkai/PycharmProjects/Datasets')
    parser.add_argument('-ds', '--degradedtest', help='o, bg, bm, hi, ho, ns, nsp', default='o')
    args = parser.parse_args()

    test_model = args.model
    gpu = args.gpu
    dataset = args.dataset
    dataset_root = args.datasetroot
    degradedtest = args.degradedtest
    out = get_log_test_dir('fcn8s-atonce', dataset, degradedtest, test_model)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
        root = osp.expanduser(osp.join(dataset_root, dataset))
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    if dataset == 'VOC':
        test_data = datasets.VOCSeg(root, split='test', dataset=degradedtest, transform=True)
    elif dataset == "CamVid":
        test_data = datasets.CamVidSeg(root, split='test', dataset=degradedtest, transform=True)
    elif dataset == "Custom":
        test_data = datasets.CustomSeg(root, split='test', dataset=degradedtest, transform=True)
    elif dataset == "Cityscapes":
        test_data = datasets.CityscapesSeg(root, split='test', dataset=degradedtest, transform=True)
    else:
        test_data = datasets.SUNSeg(root, split='test', dataset=degradedtest, transform=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, **kwargs)

    # 2. model
    model = models.FCN8sAtOnce(n_class=test_data.n_classes)
    if test_model:
        test_model = torch.load(test_model)
        model.load_state_dict(test_model['model_state_dict'])
    else:
        raise ValueError('test model is not provided')

    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)

    # 3. tester
    tester = Tester(
        cuda=cuda,
        model=model,
        test_data=test_data,
        test_loader=test_loader,
        out=out,
    )
    tester.test()


if __name__ == '__main__':
    main()
