# Chainer-DCGAN-MNIST
A Chainer implementation of Generative Adversarial Networks on the MNIST dataset.

Some of the techniques presented in the paper "Improved Techniques for Training GANs" https://arxiv.org/abs/1606.03498 are used.

There is no fully-connected layer and no pooling layer.


## Usage
    python gan_mnist.py -g 0 -b 150 -e 200
If there is no GPU, specify `-g -1`.

## Requirement
Chainer v1.15.0+
