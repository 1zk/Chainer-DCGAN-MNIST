# ^-^ coding: UTF-8 ^-^
import argparse

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer import Link, Chain, ChainList
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from scipy.misc import imsave


class Generator(Chain):

    def __init__(self, z_dim):
        super(Generator, self).__init__(
            # in_ch,out_ch,ksize,stride,pad
            l1=L.Deconvolution2D(z_dim, 128, 3, 2, 0),
            bn1=L.BatchNormalization(128),
            l2=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn2=L.BatchNormalization(128),
            l3=L.Deconvolution2D(128, 128, 3, 2, 1),
            bn3=L.BatchNormalization(128),
            l4=L.Deconvolution2D(128, 128, 3, 2, 2),
            bn4=L.BatchNormalization(128),
            l5=L.Deconvolution2D(128, 1, 3, 2, 2, outsize=(28, 28)),
        )

    def __call__(self, z):
        z = F.reshape(z, (z.data.shape[0], -1, 1, 1))
        h = self.bn1(F.relu(self.l1(z)))
        h = self.bn2(F.relu(self.l2(h)))
        h = self.bn3(F.relu(self.l3(h)))
        h = self.bn4(F.relu(self.l4(h)))
        x = F.sigmoid(self.l5(h))
        return x


class Discriminator(Chain):

    def __init__(self):
        super(Discriminator, self).__init__(
            # in_ch,out_ch,ksize,stride,pad
            l1=L.Convolution2D(None, 32, 3, 2, 1),
            bn1=L.BatchNormalization(32),
            l2=L.Convolution2D(None, 32, 3, 2, 2),
            bn2=L.BatchNormalization(32),
            l3=L.Convolution2D(None, 32, 3, 2, 1),
            bn3=L.BatchNormalization(32),
            l4=L.Convolution2D(None, 32, 3, 2, 1),
            bn4=L.BatchNormalization(32),
            l5=L.Convolution2D(None, 1, 3, 2, 1),
        )

    def __call__(self, x):
        h = self.bn1(F.leaky_relu(self.l1(x)))
        h = self.bn2(F.leaky_relu(self.l2(h)))
        h = self.bn3(F.leaky_relu(self.l3(h)))
        h = self.bn4(F.leaky_relu(self.l4(h)))
        y = F.flatten(self.l5(h))
        return y


class GAN_Updater(training.StandardUpdater):

    def __init__(self, iterator, generator, discriminator, optimizers, batchsize,
                 converter=convert.concat_examples, device=None, z_dim=2,):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.gen = generator
        self.dis = discriminator
        self._optimizers = optimizers
        self.converter = converter
        self.device = device

        self.iteration = 0

        self.z_dim = z_dim
        self.batchsize = batchsize

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        x_data = in_arrays
        z = Variable(cuda.cupy.random.normal(
            size=(self.batchsize, self.z_dim), dtype=np.float32))
        global x_gen
        x_gen = self.gen(z)

        # concatしないままdisに通すと、bnが悪さをする
        x = F.concat((x_gen, x_data), 0)
        y = self.dis(x)
        y_gen, y_data = F.split_axis(y, 2, 0)

        # sigmoid_cross_entropy(x, 0) == softplus(x)
        # sigmoid_cross_entropy(x, 1) == softplus(-x)
        loss_gen = F.sum(F.softplus(-y_gen)) / self.batchsize
        loss_data = F.sum(F.softplus(y_data)) / self.batchsize
        loss = loss_gen + loss_data

        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

        # compute gradients all at once
        loss.backward()

        for optimizer in self._optimizers.values():
            optimizer.update()

        reporter.report(
            {'loss': loss, 'loss_gen': loss_gen, 'loss_data': loss_data})


def save_x(x_gen):
    x_gen_img = cuda.to_cpu(x_gen.data)
    n = x_gen_img.shape[0]
    n = n // 15 * 15
    x_gen_img = x_gen_img[:n]
    x_gen_img = x_gen_img.reshape(
        15, -1, 28, 28).transpose(1, 2, 0, 3).reshape(-1, 15 * 28)
    imsave('x_gen.png', x_gen_img)


def main():
    parser = argparse.ArgumentParser(description='GAN_MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--z_dim', '-z', default=2,
                        help='Dimension of random variable')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# z_dim: {}'.format(args.z_dim))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    gen = Generator(args.z_dim)
    dis = Discriminator()
    gen.to_gpu()
    dis.to_gpu()

    opt = {'gen': optimizers.Adam(alpha=-0.001, beta1=0.5),  # alphaの符号が重要
           'dis': optimizers.Adam(alpha=0.001, beta1=0.5)}
    opt['gen'].setup(gen)
    opt['dis'].setup(dis)

    train, test = datasets.get_mnist(withlabel=False, ndim=3)

    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize)

    updater = GAN_Updater(train_iter, gen, dis, opt,
                          batchsize=args.batchsize, device=args.gpu, z_dim=args.z_dim)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.dump_graph('loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'loss', 'loss_gen', 'loss_data']))
    trainer.extend(extensions.ProgressBar(update_interval=100))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    np.save('x_gen.npy', cuda.to_cpu(x_gen.data))
    save_x(x_gen)


if __name__ == '__main__':
    main()
