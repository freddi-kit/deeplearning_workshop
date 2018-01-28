import sys,os
from PIL import Image
from network import Network
from chainer import Variable,optimizers
import numpy as np
import random
from chainer.datasets import mnist



batch = 10

argv = sys.argv

input_size = int(argv[1])

network_sizes = []

for i in argv[2].split(','):
    network_sizes.append(int(i))

dir_train = argv[3]+'/'
dir_lists = os.listdir(dir_train)

epoch = 1000

net = Network(network_sizes,len(dir_lists))
optimizer = optimizers.SGD()
optimizer.setup(net)


train_data = []
train_label = []

test_data = []
test_label = []

for i in dir_lists:
    sub_dirs = os.listdir(argv[3]+'/'+i+'/')
    for j in np.random.permutation(range(len(sub_dirs))):
        img = Image.open(argv[3]+'/'+i+'/'+sub_dirs[j])
        img = img.resize((input_size,input_size)).convert('RGB')
        img = np.asarray(img,dtype=np.float32).transpose((2,0,1))/255.
        train_data += [img]
        train_label += [dir_lists.index(i)]
'''
train, test = mnist.get_mnist(withlabel=True, ndim=2)


for i in range(len(train)):
    m = 0
    for j in np.random.permutation(range(len(train))):
        img = np.asarray([train[j][0]],dtype=np.float32)
        train_data += [img]
        train_label += [np.asarray(train[j][1])]
        m+=1
        if m >= max_train:
            break
'''

for e in range(epoch):
    train_data_sub = []
    train_label_sub = []
    for i in np.random.permutation(range(len(train_data))):
        train_data_sub += [train_data[i]]
        train_label_sub += [train_label[i]]
    print('epoch',e)

    for i in range(0,len(train_data_sub),batch):
        x = Variable(np.asarray(train_data_sub[i:i+batch],dtype=np.float32))
        t = Variable(np.asarray(train_label_sub[i:i+batch]))
        y,loss = net(x,t)
        net.cleargrads()
        loss.backward()
        optimizer.update()

        print(loss.data)
