from options.train_options import TrainOptions
import os
import functools

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, LeakyReLU, Softmax
import numpy as np

opt = TrainOptions().parse()

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class SpectralDiscriminator(tf.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=InstanceNormalization, use_sigmoid=False):
        super(SpectralDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != InstanceNormalization
        else:
            use_bias = norm_layer != InstanceNormalization

        kw = 4
        padw = 1
        sequence = [SpectralNormalization(Conv2D(input_shape=(None, None, input_nc), kernel_size=kw, stride=2, padding='valid')),
                    LeakyReLU(alpha=0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SpectralNormalization(Conv2D(filters = ndf *
                              nf_mult, kernel_size=kw, stride=2, padding=padw)),
                # norm_layer(ndf * nf_mult),
                LeakyReLU(alpha=0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SpectralNormalization(Conv2D(filters= ndf *
                          nf_mult, kernel_size=kw, stride=1, padding=padw)),
            # norm_layer(ndf * nf_mult),
            LeakyReLU(0.2, True)
        ]

        sub_sequence = [SpectralNormalization(Conv2D(ndf * nf_mult,
                                   1, kernel_size=kw, stride=1, padding=padw))]       
        sequence += sub_sequence       
        if use_sigmoid:
            sequence += [tf.keras.activations.sigmoid(sub_sequence)]
        self.model = Sequential(*sequence)
        self.old_lr = opt.lr_D

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    def update_learning_rate(self, optimizer, opt):
        lrd = opt.lr_D / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.local_rank == 0:
            print('update learning rate for D model: %f -> %f' %
                  (self.old_lr, lr))
        self.old_lr = lr


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(tf.Module):
    def __init__(self, use_lsgan=True, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.real_label = tf.Variable(tf.convert_to_tensor(target_real_label), trainable=False)
        self.fake_label = tf.Variable(tf.convert_to_tensor(target_fake_label), trainable=False)
        if use_lsgan:
            self.loss = tf.keras.losses.MeanSquaredError()
        else:
            self.loss = tf.keras.losses.BinaryCrossentropy()
        assert gan_mode in ['lsgan', 'vanilla', 'wgangp']
        if gan_mode in ['wgangp']:
            self.loss = None
        self.gan_mode = gan_mode

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        input_shape = tf.shape(input)

        # Mở rộng target_tensor để có cùng kích thước với tensor đầu vào
        return tf.broadcast_to(target_tensor, input_shape)

    def __call__(self, prediction, target_is_real, add_gradient=False):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()  # + 0.001*(prediction**2).mean()
                if add_gradient:
                    loss = -prediction.mean() + 0.001*(prediction**2).mean()
            else:
                loss = prediction.mean()
        return loss



class ResidualBlock(tf.Module):
    def __init__(self, in_features=64, norm_layer=tf.keras.layers.BatchNormalization()):
        super(ResidualBlock, self).__init__()
        self.relu = tf.keras.activations.relu()
        if norm_layer == None:
            self.block = Sequential(
                Conv2D(in_features, in_features, 3, 1, 1, bias=False),
                tf.keras.activations.relu(inplace=True),
                Conv2D(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = Sequential(
                Conv2D(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                tf.keras.activations.relu(inplace=True),
                Conv2D(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResUnetGenerator(tf.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=tf.keras.layers.BatchNormalization(), use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block
        self.old_lr = opt.lr
        self.old_lr_gmm = 0.1*opt.lr

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class ResUnetSkipConnectionBlock(tf.keras.Model):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=tf.keras.layers.BatchNormalization(), use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == InstanceNormalization()

        if input_nc is None:
            input_nc = outer_nc
        downconv = tf.keras.layers.Conv2D(inner_nc, kernel_size=3, strides=2, padding='same', use_bias=use_bias)
        # add two resblock
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = tf.keras.activations.relu()
        uprelu = tf.keras.activations.relu()
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = tf.keras.layers.UpSampling2D(size=2)
            upconv = tf.keras.layers.Conv2D(outer_nc, kernel_size=3, strides=1, padding='same', use_bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = tf.keras.layers.UpSampling2D(size=2)
            upconv = tf.keras.layers.Conv2D(outer_nc, kernel_size=3, strides=1, padding='same', use_bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = tf.keras.layers.UpSampling2D(size=2)
            upconv = tf.keras.layers.Conv2D(outer_nc, kernel_size=3, strides=1, padding='same', use_bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [tf.keras.layers.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = tf.keras.Sequential(model)

    def call(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return tf.concat([x, self.model(x)], axis=-1)


class Vgg19(tf.keras.Model):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # Tải mô hình VGG19 được đào tạo trước
        vgg_pretrained = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        
        # Tạo các lớp
        self.slice1 = tf.keras.Sequential(vgg_pretrained.layers[:2])
        self.slice2 = tf.keras.Sequential(vgg_pretrained.layers[2:7])
        self.slice3 = tf.keras.Sequential(vgg_pretrained.layers[7:12])
        self.slice4 = tf.keras.Sequential(vgg_pretrained.layers[12:21])
        self.slice5 = tf.keras.Sequential(vgg_pretrained.layers[21:30])
        
        # Đặt requires_grad
        if not requires_grad:
            for layer in self.layers:
                layer.trainable = False

    def call(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(tf.keras.Model):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = tf.keras.losses.MeanAbsoluteError()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def call(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    tf.keras.models.save_model(model, save_path)


def load_checkpoint_parallel(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return

    # Tạo một đối tượng Checkpoint
    checkpoint = tf.train.Checkpoint(model=model)

    # Tải trạng thái từ checkpoint
    checkpoint.restore(checkpoint_path)

# Hàm này chỉ hỗ trợ trong pytorch
def load_checkpoint_part_parallel(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    # Kiểm tra phần tải weight
    checkpoint = tf.keras.models.load_model(checkpoint_path,map_location='cuda:{}'.format(opt.local_rank))
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        if 'cond_' not in param and 'aflow_net.netRefine' not in param:
            checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


class NetworkBase(tf.keras.Model):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(tf.keras.layers.BatchNormalization())
        elif norm_type == 'instance':
            # InstanceNormalization không có sẵn trong tf.keras.layers
            # Bạn cần tự định nghĩa hoặc tìm một thư viện hỗ trợ
            norm_layer = functools.partial(InstanceNormalization())
        elif norm_type == 'batchnorm2d':
            norm_layer = tf.keras.layers.BatchNormalization()
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer

    def call(self, *input):
        raise NotImplementedError
