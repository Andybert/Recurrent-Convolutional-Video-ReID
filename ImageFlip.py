import numpy as np


def flip(img, mode):
    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]
    image_flip = np.array(np.zeros((height, width, channel), dtype=np.float32))
    if mode == 'Left2Right':
        for w in xrange(0, width):
            w_index = width - w - 1
            for h in xrange(0, height):
                for c in xrange(0, channel):
                    image_flip[h, w_index, c] = img[h, w, c]
    elif mode == 'Up2Down':
        for h in xrange(0, height):
            h_index = height - h - 1
            for w in xrange(0, width):
                for c in xrange(0, channel):
                    image_flip[h_index, w, c] = img[h, w, c]
    else:
        raise Exception('flip type error!)')
    return image_flip


def crop(rawFrame, scropx, scropy, ecropx, ecropy):
    # print rawFrame.shape
    # print scropx, scropy, ecropx, ecropy
    frame = np.zeros((5, 56, 40), dtype=np.float32)
    frame = rawFrame[:, scropy:ecropy, scropx:ecropx]
    return frame
