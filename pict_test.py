import sys,os
from PIL import Image
from network import Network
from chainer import Variable,serializers,cuda
from chainer import functions as F
import numpy as np

import random
from chainer.datasets import mnist

GPU = -1

batch = 10
argv = sys.argv
ixput_size = int(argv[1])

network_sizes = []

for i in argv[2].split(','):
    network_sizes.append(int(i))

dir_train = argv[4]+'/'
dir_lists = os.listdir(dir_train)

net = Network(network_sizes,len(dir_lists))
serializers.load_npz(argv[3],net)

if GPU >= 0:
    gpu_device = 0
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
    sub_dirs = os.listdir(argv[4]+'/'+i+'/')
    for j in xp.random.permutation(range(len(sub_dirs))):
        img = Image.open(argv[4]+'/'+i+'/'+sub_dirs[j])
        img = img.resize((ixput_size,ixput_size)).convert('RGB')
        img = xp.asarray(img,dtype=xp.float32).transpose((2,0,1))/255.
        test_data += [img]
        test_label += [dir_lists.index(i)]

acc=0
b=0
for i in range(0,len(test_data),batch):
    x = Variable(xp.asarray(test_data[i:i+batch],dtype=xp.float32))
    t = Variable(xp.asarray(test_label[i:i+batch]))
    y = net.predict(x)
    accuracy = F.accuracy(y, t)
    accuracy.to_cpu()
    acc+=accuracy.data
    b+=1

print((acc/b)*100,'%')
