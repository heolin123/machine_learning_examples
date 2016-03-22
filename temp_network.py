# -*- coding: utf-8 -*-
from pybrain.structure import FeedForwardNetwork, LinearLayer, FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from scipy.signal import firwin, lfilter
import numpy as np
import pylab as py


def generujAR(a, epsilon, N):
    x=np.zeros(N)
    rzad = len(a)
    for i in range(rzad,N):
        for p in range(len(a)):
            x[i] += a[p]*x[i-(p+1)]
        x[i] += epsilon*np.random.randn()
    return x

N = 2000
x = np.zeros(N)
a = np.array([ -0.5, 0.2])
epsilon = 0.1
print "GENERUJE AR"
x = generujAR(a,epsilon,N)
py.plot(x)

print "TWORZY CIAG UCZACY"
# tworzymy ciąg uczący CU
N_wej = 2
CU = SupervisedDataSet(N_wej, 1)
for i in range(len(x)-N_wej-1):
    bufor_wejsciowy = x[i:i+N_wej]
    wartosc_wyjsciowa = x[i+N_wej]
    CU.addSample( bufor_wejsciowy[::-1], wartosc_wyjsciowa)#
    print  bufor_wejsciowy[::-1], wartosc_wyjsciowa

print CU

print "TWORZY PUSTA SIEC"
# wytwarzamy pustą sieć
siec = FeedForwardNetwork()

# tworzymy węzły wejściowe i wyjściowe

warstwaWejsciowa = LinearLayer(N_wej)
warstwaWyjsciowa = LinearLayer(1)

# dodajemy węzły do sieci
siec.addInputModule(warstwaWejsciowa)
siec.addOutputModule(warstwaWyjsciowa)

# łączymy węzły
wej_do_wyj = FullConnection(warstwaWejsciowa, warstwaWyjsciowa)
siec.addConnection(wej_do_wyj)

# inicjujemy strukturę sieci
siec.sortModules()

print "ROZPOCZYNA UCZENIE"

trainer = BackpropTrainer(siec, CU,learningrate=0.01, momentum=0.8,verbose=True)
#trainer.trainUntilConvergence(maxEpochs = 500)
for i in range(10):
    print i
    trainer.trainEpochs(1)
    print i, siec.params

# test na zbiorze uczącym
print "TEST NA ZBIORZE UCZACYM"
y_est = np.zeros(x.shape)
for i in range(len(x)-N_wej-1):
    bufor_wejsciowy = x[i:i+N_wej]
    y_est[i+N_wej] = siec.activate(bufor_wejsciowy[::-1])
#py.plot(t,x)
py.subplot(2,1,1)
py.plot(x,'b')
py.plot(y_est,'r')
py.subplot(2,1,2)
py.plot(x-y_est)
print 'wariancja reziduum, ', np.var(x-y_est)
print 'wariancja procesu AR, ', epsilon**2
print 'parametry procesu: ', str(a)
print 'parametry wyestymowane: ', str(siec.params)
py.show()
