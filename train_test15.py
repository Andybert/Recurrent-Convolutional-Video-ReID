import torch
# import torchvision
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as Func
import torch.optim as optim
# import os
# import argparse
# from torch.utils.serialization import load_lua
import random
from buildFullModel11 import FullModel
from dataloader11 import DataLoader

# os.system('script -f  history9.log')
RGB_root = '/mnt/68FC8564543F417E/Data/i-LIDS-VID/sequences/'
OF_root = '/mnt/68FC8564543F417E/Data/i-LIDS-VID-OF-HVP/sequences/'
ranknum = 20
train_test_rate = 0.5
sampleSeqLength = 16
nFilters = [16, 32, 32]
opt = {'clip': 5, 'lr': 0.01, 'dropoutFrac': 0.6, 'dropoutFracRNN': 0.6, 'embeddingSize': 128, 'sampleSeqLen': sampleSeqLength, 'scale_num': 4, 'dataset': 1}
dataloader = DataLoader(opt)
if opt['dataset'] == 1:
    seqRootRGB = '/mnt/68FC8564543F417E/Data/i-LIDS-VID/sequences/'
    seqRootOF = '/mnt/68FC8564543F417E/Data/i-LIDS-VID-OF-HVP/sequences/'
else:
    seqRootRGB = '/mnt/68FC8564543F417E/Data/PRID2011/multi_shot/'
    seqRootOF = '/mnt/68FC8564543F417E/Data/PRID2011-OF-HVP/multi_shot/'
print 'loading Dataset - ', seqRootRGB, seqRootOF
person_list, dataset = dataloader.prepareDataset(seqRootRGB, seqRootOF)
trainList, testList = dataloader.partitionDataset(person_list, 0.5)
nTrainPersons = len(trainList)
nTestPersons = len(testList)
opt['train_category_num'] = nTrainPersons
model = FullModel(nFilters, opt)
model.cuda()

criterion_pair = nn.HingeEmbeddingLoss(margin=2.0, size_average=True)
criterion_soft1 = nn.CrossEntropyLoss(weight=None, size_average=True)
criterion_soft2 = nn.CrossEntropyLoss(weight=None, size_average=True)
optimizer = optim.SGD(model.parameters(), lr=opt['lr'])

# # in your training loop:

epochs = 1000


def train(epoch):
    model.train()
    order = range(0, nTrainPersons)
    np.random.shuffle(order)
    totalloss = 0.0
    for i in xrange(0, nTrainPersons):
        # print 'iter: ', i
        if i % 2 == 0:
            seqA, seqB, camA, camB, startA, startB, seq_length = dataloader.getPosSample(dataset, trainList, order[i], opt['sampleSeqLen'])
            imagePixelDataA = dataset[seqA][camA]['data'][startA:startA + seq_length, :, :, :]
            imagePixelDataB = dataset[seqB][camB]['data'][startB:startB + seq_length, :, :, :]
            pair_labels = Variable(torch.FloatTensor([1])).cuda()
            soft_labelsA = Variable(torch.LongTensor([trainList.index(seqA)])).cuda()
            soft_labelsB = Variable(torch.LongTensor([trainList.index(seqB)])).cuda()
        else:
            seqA, seqB, camA, camB, startA, startB, seq_length = dataloader.getNegSample(dataset, trainList, opt['sampleSeqLen'])
            imagePixelDataA = dataset[seqA][camA]['data'][startA:startA + seq_length, :, :, :]
            imagePixelDataB = dataset[seqB][camB]['data'][startB:startB + seq_length, :, :, :]
            pair_labels = Variable(torch.FloatTensor([-1])).cuda()
            soft_labelsA = Variable(torch.LongTensor([trainList.index(seqA)])).cuda()
            soft_labelsB = Variable(torch.LongTensor([trainList.index(seqB)])).cuda()

        crpxA = random.randint(0, 7)
        crpyA = random.randint(0, 7)
        crpxB = random.randint(0, 7)
        crpyB = random.randint(0, 7)
        flipA = random.randint(0, 1)
        flipB = random.randint(0, 1)
        netInputA = Variable(dataloader.doDataAug(imagePixelDataA, crpxA, crpyA, flipA)).cuda()
        netInputB = Variable(dataloader.doDataAug(imagePixelDataB, crpxB, crpyB, flipB)).cuda()
        optimizer.zero_grad()   # zero the gradient buffers

        # print 'netInputA:\n', netInputA
        featuresA, featuresB, soft_out1, soft_out2, pair_out = model(netInputA, netInputB)
        pair_loss = criterion_pair(pair_out, pair_labels)
        soft_loss1 = criterion_soft1(soft_out1, soft_labelsA)
        soft_loss2 = criterion_soft2(soft_out2, soft_labelsB)
        loss = pair_loss + soft_loss1 + soft_loss2
        totalloss = totalloss + loss

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opt['clip'])
        # for p in model.parameters():
        #     p.data.add_(-opt['lr'], p.grad.data)
        optimizer.step()
    print 'epoch %4d loss: %f' % (epoch, totalloss.cpu().data.numpy()[0])


def cmc_compute(test_list, seq_len):
    test_num = len(test_list)
    dist_matrix = np.zeros((test_num, test_num), dtype=np.float64)
    for cr in xrange(0, 8):
        for fp in xrange(0, 2):
            for i in xrange(0, test_num):
                # seqA, seqB, camA, camB, startA, startB, seq_length = dataloader.getPosSample(dataset, test_list, i, seq_len)
                if opt['dataset'] == 1:
                    camA = "cam1"
                    camB = "cam2"
                else:
                    camA = "cam_a"
                    camB = "cam_b"
                actualSampleSeqLenA = min(dataset[test_list[i]][camA]['frames_num'], seq_len)
                actualSampleSeqLenB = min(dataset[test_list[i]][camB]['frames_num'], seq_len)
                imagePixelDataA = dataset[test_list[i]][camA]['data'][0:actualSampleSeqLenA, :, :, :]
                imagePixelDataB = dataset[test_list[i]][camB]['data'][-actualSampleSeqLenB:, :, :, :]
                netInputA = Variable(dataloader.doDataAug(imagePixelDataA, cr, cr, fp)).cuda()
                netInputB = Variable(dataloader.doDataAug(imagePixelDataB, cr, cr, fp)).cuda()
                featuresA, featuresB, _, _, _ = model(netInputA, netInputB)
                if i == 0:
                    features_camA = np.zeros((test_num, featuresA.cpu().data.numpy()[0].shape[0]))
                    features_camB = np.zeros((test_num, featuresA.cpu().data.numpy()[0].shape[0]))
                features_camA[i, :] = featuresA.cpu().data.numpy()[0]
                features_camB[i, :] = featuresB.cpu().data.numpy()[0]

            for i in xrange(0, test_num):
                for j in xrange(0, test_num):
                    dist_matrix[i, j] = dist_matrix[i, j] + np.linalg.norm(features_camA[i, :] - features_camB[j, :])
    acc = np.zeros(ranknum)
    tp = 0
    sorted_index = np.argsort(dist_matrix, 0)
    for r in xrange(0, ranknum):
        tp = 0
        for i in xrange(0, test_num):
            temp = sorted_index[0:r + 1, i]
            if i in temp:
                tp = tp + 1
        acc[r] = tp / float(test_num)
    return acc


def test(epoch):
    model.eval()
    sl_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for sl in sl_list:
        acc = cmc_compute(testList, sl)
        print 'epoch %d test %d images, test accuracy:\n' % (epoch, sl)
        print acc


for epoch in range(1, epochs + 1):
    train(epoch)
    if (epoch % 100 == 0):
        test(epoch)
    torch.save(model, '/mnt/68FC8564543F417E/Pytorch/model10.pt')
# os.system('exit')
