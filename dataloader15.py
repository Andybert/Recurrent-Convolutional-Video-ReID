import sys
from skimage import io as sio
from skimage.transform import resize
from skimage.util import crop
from skimage.color import rgb2yuv
import matplotlib.pyplot as plt
import scipy.misc
import time
import glob
import random
import os
import torch
import numpy as np
import copy
from ImageFlip import flip as Iflip
from ImageFlip import crop as Icrop
from torchvision import transforms

transform_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Scale(48, 64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_aug = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(40, 56),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def Programpause():
    inputstr = raw_input('Press e to exit, Press other key to continue!!!\n')
    if (inputstr == 'e'):
        os._exit(0)


class DataLoader(object):
    """docstring for prepareDataset"""

    def __init__(self, opt):
        super(DataLoader, self).__init__()
        self.opt = opt

    def loadSequenceImages(self, cameraDir, opticalflowDir, filesList):
        nImgs = len(filesList)
        for i, file in enumerate(filesList):
            filename = cameraDir + '/' + file
            filenameOF = opticalflowDir + '/' + file

            img = sio.imread(filename) * 1.0
            img = rgb2yuv(resize(img, (64, 48), mode='reflect'))

            imgof = sio.imread(filenameOF) * 1.0
            imgof = resize(imgof, (64, 48), mode='reflect')

            if i == 0:
                s = img.shape
                imagePixelData = np.zeros((nImgs, 5, s[0], s[1]), dtype=np.float64)

            for c in xrange(0, 3):
                d = img[:, :, c]
                d = d
                m = np.mean(d, axis=(0, 1))
                v = np.std(d, axis=(0, 1))
                d = d - m
                d = d / v
                imagePixelData[i, c, :, :] = d
            for c in xrange(0, 2):
                d = imgof[:, :, c]
                d = d
                m = np.mean(d, axis=(0, 1))
                v = np.std(d, axis=(0, 1))
                d = d - m
                d = d / v
                imagePixelData[i, c + 3, :, :] = d
        return imagePixelData

    def getSequenceImageFiles(self, seqRoot):
        seqFiles = []
        seqRoot = os.path.expanduser(seqRoot)
        for file in os.listdir(seqRoot):
            seqFiles.append(file)
        seqFiles = sorted(seqFiles, key=lambda pd: int(pd.split('.')[0].split('_')[-1]))
        return seqFiles

    def getPersonDirsList(self, seqRootDir):
        if self.opt['dataset'] == 1:
            firstCameraDirName = 'cam1'
        else:
            firstCameraDirName = 'cam_a'
        tmpSeqCam = seqRootDir + '/' + firstCameraDirName

        personDirs = []
        for file in os.listdir(tmpSeqCam):
            if len(file) > 2:
                personDirs.append(file)

        if len(personDirs) == 0:
            raise Exception(seqRootDir + ' directory does not contain any image files')

        if self.opt['dataset'] == 1:
            delimiter = "n"
        else:
            delimiter = "_"
        j = personDirs[0].index(delimiter)
        personDirs = sorted(personDirs, key=lambda pd: int(pd[j + 1:]))
        return personDirs

    def prepareDataset(self, datasetRootDir, datasetRootDirOF):
        dataset = {}
        personDirs = self.getPersonDirsList(datasetRootDir)
        nPersons = len(personDirs)
        letter = ['a', 'b']
        for i in range(0, nPersons):
            if dataset.has_key(personDirs[i]) is False:
                dataset[personDirs[i]] = {}
            for cam in xrange(1, 3):
                if self.opt['dataset'] == 1:
                    cameraDirName = "cam" + str(cam)
                else:
                    cameraDirName = 'cam_' + letter[cam]
                seqRoot = datasetRootDir + cameraDirName + '/' + personDirs[i]
                seqRootOF = datasetRootDirOF + cameraDirName + '/' + personDirs[i]
                seqImgs = self.getSequenceImageFiles(seqRoot)
                if dataset[personDirs[i]].has_key(cameraDirName) is False:
                    dataset[personDirs[i]][cameraDirName] = {}
                dataset[personDirs[i]][cameraDirName]['frames_num'] = len(seqImgs)
                dataset[personDirs[i]][cameraDirName]['data'] = self.loadSequenceImages(seqRoot, seqRootOF, seqImgs)
                if len(dataset[personDirs[i]][cameraDirName]) == 0:
                    raise Exception('no dimension')

        return personDirs, dataset

    def partitionDataset(self, person_list, testTrainSplit):
        nTotalPersons = len(person_list)
        splitPoint = int(np.floor(nTotalPersons * testTrainSplit))
        inds = range(0, nTotalPersons)
        np.random.shuffle(inds)

        trainList = []
        testList = []
        for x in xrange(0, splitPoint):
            trainList.append(person_list[inds[x]])
        for x in xrange(splitPoint, nTotalPersons):
            testList.append(person_list[inds[x]])
        print 'N train = ', len(trainList)
        print 'N test = ', len(testList)

        f1 = open('dataSplit.txt', 'w')
        f1.write('train:\n')
        for x in trainList:
            f1.write(str(x))
            f1.write('\n')
        f1.write('\ntest:\n')
        for x in testList:
            f1.write(str(x))
            f1.write('\n')
        f1.close()

        return trainList, testList

    def getPosSample(self, dataset, trainList, person_index, sampleSeqLen):
        if self.opt['dataset'] == 1:
            camA = "cam1"
            camB = "cam2"
        else:
            camA = "cam_a"
            camB = "cam_b"

        # actualSampleSeqLen = sampleSeqLen
        personA = trainList[person_index]
        personB = trainList[person_index]
        nSeqA = dataset[personA][camA]['frames_num']
        nSeqB = dataset[personB][camB]['frames_num']

        actualSampleSeqLen = min(nSeqA, nSeqB, sampleSeqLen)
        # if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
        #     if nSeqA < nSeqB:
        #         actualSampleSeqLen = nSeqA
        #     else:
        #         actualSampleSeqLen = nSeqB

        startA = random.randint(0, nSeqA - actualSampleSeqLen)
        startB = random.randint(0, nSeqB - actualSampleSeqLen)

        return personA, personB, camA, camB, startA, startB, int(actualSampleSeqLen)

    def getNegSample(self, dataset, trainList, sampleSeqLen):
        permAllPersons = range(0, len(trainList))
        np.random.shuffle(permAllPersons)
        personA = trainList[permAllPersons[1]]
        personB = trainList[permAllPersons[2]]

        A = random.randint(1, 2)
        B = random.randint(1, 2)
        letter = ['_a', '_b']
        if self.opt['dataset'] == 1:
            camA = "cam" + str(A)
            camB = "cam" + str(B)
        else:
            camA = "cam" + letter[A]
            camB = "cam" + letter[B]

        # actualSampleSeqLen = sampleSeqLen
        nSeqA = dataset[personA][camA]['frames_num']
        nSeqB = dataset[personB][camB]['frames_num']

        actualSampleSeqLen = min(nSeqA, nSeqB, sampleSeqLen)
        # if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
        #     if nSeqA < nSeqB:
        #         actualSampleSeqLen = nSeqA
        #     else:
        #         actualSampleSeqLen = nSeqB

        startA = random.randint(0, nSeqA - actualSampleSeqLen)
        startB = random.randint(0, nSeqB - actualSampleSeqLen)

        return personA, personB, camA, camB, startA, startB, int(actualSampleSeqLen)

    def doDataAug(self, seq, cropx, cropy, flip):
        seqLen = seq.shape[0]
        seqChnls = seq.shape[1]
        seqDim1 = seq.shape[2]
        seqDim2 = seq.shape[3]

        daData = torch.zeros(seqLen, seqChnls, seqDim1 - 8, seqDim2 - 8)
        for t in xrange(0, seqLen):
            thisFrame = seq[t, :, :, :]
            if flip == 1:
                thisFrame = Iflip(thisFrame, 'Left2Right')

            cropimg = Icrop(thisFrame, cropx, cropy, 40 + cropx, 56 + cropy)
            m = np.mean(cropimg, axis=(0, 1, 2))
            cropimg = cropimg - m
            daData[t, :, :, :] = torch.from_numpy(cropimg)
        return daData
