import sys,os
from PIL import Image
from network import Network
from chainer import Variable,optimizers,serializers,cuda
from chainer import functions as F
import numpy as np

import random
from chainer.datasets import mnist

GPU = -1

batch = 5
argv = sys.argv
input_size = int(argv[1])

network_sizes = []

for i in argv[2].split(','):
    network_sizes.append(int(i))

dir_train = argv[3]+'/'
dir_lists = sorted(os.listdir(dir_train))

epoch = 500

net = Network(network_sizes,len(dir_lists))
optimizer = optimizers.SGD()
optimizer.setup(net)

if GPU >= 0:
    cuda.get_device(GPU).use()
    net.to_gpu(GPU)
    xp = cuda.cupy


else:
    xp = np


train_data = []
train_label = []

test_data = []
test_label = []

for i in dir_lists:
    sub_dirs = os.listdir(argv[3]+'/'+i+'/')
    for j in xp.random.permutation(range(len(sub_dirs))):
        img = Image.open(argv[3]+'/'+i+'/'+sub_dirs[j])
        img = img.resize((input_size,input_size)).convert('RGB')
        img = xp.asarray(img,dtype=xp.float32).transpose((2,0,1))/255.
        train_data += [img]
        train_label += [dir_lists.index(i)]


for e in range(epoch):
    train_data_sub = []
    train_label_sub = []

    for i in xp.random.permutation(range(len(train_data))):
        train_data_sub += [train_data[i]]
        train_label_sub += [train_label[i]]
    print('epoch',e)

    for i in range(0,len(train_data_sub),batch):
        x = Variable(xp.asarray(train_data_sub[i:i+batch],dtype=xp.float32))
        t = Variable(xp.asarray(train_label_sub[i:i+batch]))
        y,loss = net(x,t)
        net.cleargrads()
        loss.backward()
        optimizer.update()

        print(loss.data)
    if e % 10 == 0:
        serializers.save_npz('model/model.npz',net)
