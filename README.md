# Chainer-DCGAN-MNIST
A Chainer implementation of Deep Convolutional Generative Adversarial Networks on the MNIST dataset.

Some of the techniques presented in the paper "Improved Techniques for Training GANs" https://arxiv.org/abs/1606.03498 are used.

<img src="/x_gen_example_.png"/>
## Usage
    python gan_mnist.py -g 0 -b 150 -e 200
If there is no GPU, this program does not work.
## Requirement
Chainer v1.15.0+
