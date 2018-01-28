import chainer
import chainer.links as L
import chainer.functions as F
import math

class Network(chainer.Chain):
    def __init__(self,sizes,output):
        super(Network, self).__init__()
        w  = chainer.initializers.HeNormal()
        links=[]
        c_k,c_s,c_p = 3,3,1
        self.m_k,self.m_s,self.m_p = 3,1,1
        for i in range(len(sizes)):
            links += [('conv{}'.format(i), L.Convolution2D(sizes[i-1] if i > 0 else 3, sizes[i], c_k,c_s,c_p,initialW=w))]
        links += [('linear0',L.Linear(in_size=None,out_size=1000))]
        links += [('linear2',L.Linear(1000,output))]

        for link in links:
            self.add_link(*link)

        self.forward = links

    def __call__(self,x,t):
        h = x
        for name, f in self.forward:
            h = F.relu(f(h))
        loss = F.softmax_cross_entropy(h,t)
        return h,loss
    def predict(self,x):
        h = x
        for name, f in self.forward:
            if 'conv' in name:
                h = F.relu(f(h))
            else:
                h = F.relu(f(h))

        return F.softmax(h)
