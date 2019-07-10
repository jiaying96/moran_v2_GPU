# -*-coding:UTF-8-*-
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import tools.utils as utils
import tools.dataset as dataset
import time
from collections import OrderedDict
from models.moran import MORAN

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--valroot', required=True, help='path to dataset')  # reg_dataset/cute80_288
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=200, help='the width of the input image to network')
parser.add_argument('--targetH', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--targetW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')  # 1
parser.add_argument('--cuda', action='store_true', help='enables cuda')  # --cuda
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--MORAN', default= 'output/131000_0.7673.pth', help="path to model (to continue training)")
parser.add_argument('--alphabet', type=str,
                    default='0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$')
parser.add_argument('--sep', type=str, default=':')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')  # output/
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')  # 100
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')  # 1000
parser.add_argument('--saveInterval', type=int, default=10000, help='Interval to be displayed')  # 40000
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adadelta', action='store_true',
                    help='Whether to use adadelta (default is rmsprop)')  # --adadelta
parser.add_argument('--sgd', action='store_true', help='Whether to use sgd (default is rmsprop)')
parser.add_argument('--BidirDecoder', action='store_true', help='Whether to use BidirDecoder')  # --BidirDecoder
opt = parser.parse_args()


assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."



opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if not torch.cuda.is_available():
    assert not opt.cuda, 'You don\'t have a CUDA device.'

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


# 测试集准备
test_dataset = dataset.lmdbDataset(root=opt.valroot,
                                   transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)

# 类别数
nclass = len(opt.alphabet.split(opt.sep))
nc = 1

# 为Attention转换label
converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
# 损失函数
criterion = torch.nn.CrossEntropyLoss()

if opt.cuda:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda)
else:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder,
                  inputDataType='torch.FloatTensor', CUDA=opt.cuda)

if opt.MORAN != '':
    print('loading pretrained model from %s' % opt.MORAN)
    if opt.cuda:
        state_dict = torch.load(opt.MORAN)
    else:
        state_dict = torch.load(opt.MORAN, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        MORAN_state_dict_rename[name] = v

    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batchSize * 5)
text_rev = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
text_rev = Variable(text_rev)
length = Variable(length)

# loss averager


def val(dataset, criterion, max_iter=1000):

    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))  # opt.batchSize
    val_iter = iter(data_loader)
    max_iter = min(max_iter, len(data_loader))

    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()

    for i in range(max_iter):
        data = val_iter.next()
        if opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = MORAN(image, length, text, text_rev, test=True)
            cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
            preds0_prob, preds0 = preds0.max(1)
            preds0 = preds0.view(-1)
            preds0_prob = preds0_prob.view(-1)
            sim_preds0 = converter.decode(preds0.data, length.data)
            preds1_prob, preds1 = preds1.max(1)
            preds1 = preds1.view(-1)
            preds1_prob = preds1_prob.view(-1)
            sim_preds1 = converter.decode(preds1.data, length.data)
            sim_preds = []
            for j in range(cpu_images.size(0)):
                text_begin = 0 if j == 0 else length.data[:j].sum()
                if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).data[0] > \
                        torch.mean(preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).data[0]:
                    sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                else:
                    sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = MORAN(image, length, text, text_rev, test=True)
            cost = criterion(preds, text)
            _, preds = preds.max(1)
            preds = preds.view(-1)
            sim_preds = converter.decode(preds.data, length.data)


        # print("cpu_texts ")
        # print(cpu_texts)

        # print("预测结果")
        # print(sim_preds)

        for res in sim_preds:
            print(res.strip().split('$')[0])



        loss_avg.add(cost)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            n_total += 1

    print("correct / total: %d / %d" % (n_correct, n_total))
    accuracy = n_correct / float(n_total)
    print('accuray: %f' % accuracy)
    return accuracy





for p in MORAN.parameters():
    p.requires_grad = False
MORAN.eval()
acc_tmp = val(test_dataset, criterion)