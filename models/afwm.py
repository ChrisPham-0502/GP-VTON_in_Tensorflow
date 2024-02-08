import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, LeakyReLU, Softmax
import numpy as np

from correlation import correlation
from backup import grid_sample, interpolate

def apply_offset(offset):
    sizes = offset.shape[2:]
    grid_list = tf.meshgrid(*[tf.range(size, dtype=tf.float32) for size in sizes])

    # apply offset
    grid_list = [grid + tf.expand_dims(offset[:, dim, ...], 0)
                 for dim, grid in enumerate(grid_list)]

    # normalize
    grid_list = [(grid / ((size - 1.0) / 2.0)) - 1.0
                 for grid, size in zip(grid_list, sizes)]

    return tf.stack(grid_list, axis=-1)


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return tf.reduce_mean(tf.abs(tv_h)) + tf.reduce_mean(tf.abs(tv_w))


def TVLoss_v2(x, mask):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    h, w = mask.shape[2], mask.shape[3]

    tv_h = tv_h * mask[:, :, :h-1, :]
    tv_w = tv_w * mask[:, :, :, :w-1]

    mask_sum = tf.reduce_sum(mask)

    if mask_sum > 0:
        return (tf.reduce_sum(tf.abs(tv_h)) + tf.reduce_sum(tf.abs(tv_w))) / mask_sum
    else:
        return tf.reduce_sum(tf.abs(tv_h)) + tf.reduce_sum(tf.abs(tv_w))

def SquareTVLoss(flow):
    flow_x, flow_y = tf.split(flow, 1, dim=1)

    flow_x_diff_left = flow_x[:, :, :, 1:] - flow_x[:, :, :, :-1]
    flow_x_diff_right = flow_x[:, :, :, :-1] - flow_x[:, :, :, 1:]
    flow_x_diff_left = flow_x_diff_left[...,1:-1,:-1]
    flow_x_diff_right = flow_x_diff_right[...,1:-1,1:]

    flow_y_diff_top = flow_y[:, :, 1:, :] - flow_y[:, :, :-1, :]
    flow_y_diff_bottom = flow_y[:, :, :-1, :] - flow_y[:, :, 1:, :]
    flow_y_diff_top = flow_y_diff_top[...,:-1,1:-1]
    flow_y_diff_bottom = flow_y_diff_bottom[...,1:,1:-1]

    left_top_diff = tf.abs(tf.abs(flow_x_diff_left) - tf.abs(flow_y_diff_top))
    left_bottom_diff = tf.abs(tf.abs(flow_x_diff_left) - tf.abs(flow_y_diff_bottom))
    right_top_diff = tf.abs(tf.abs(flow_x_diff_right) - tf.abs(flow_y_diff_top))
    right_bottom_diff = tf.abs(tf.abs(flow_x_diff_right) - tf.abs(flow_y_diff_bottom))

    return tf.reduce_mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

def SquareTVLoss_v2(flow, interval_list=[1,5]):
    flow_x, flow_y = tf.split(flow, 1, dim=1)

    tvloss = 0
    for interval in interval_list:
        flow_x_diff_left = flow_x[:, :, :, interval:] - flow_x[:, :, :, :-interval]
        flow_x_diff_right = flow_x[:, :, :, :-interval] - flow_x[:, :, :, interval:]
        flow_x_diff_left = flow_x_diff_left[...,interval:-interval,:-interval]
        flow_x_diff_right = flow_x_diff_right[...,interval:-interval,interval:]

        flow_y_diff_top = flow_y[:, :, interval:, :] - flow_y[:, :, :-interval, :]
        flow_y_diff_bottom = flow_y[:, :, :-interval, :] - flow_y[:, :, interval:, :]
        flow_y_diff_top = flow_y_diff_top[...,:-interval,interval:-interval]
        flow_y_diff_bottom = flow_y_diff_bottom[...,interval:,interval:-interval]

        left_top_diff = tf.abs(tf.abs(flow_x_diff_left) - tf.abs(flow_y_diff_top))
        left_bottom_diff = tf.abs(tf.abs(flow_x_diff_left) - tf.abs(flow_y_diff_bottom))
        right_top_diff = tf.abs(tf.abs(flow_x_diff_right) - tf.abs(flow_y_diff_top))
        right_bottom_diff = tf.abs(tf.abs(flow_x_diff_right) - tf.abs(flow_y_diff_bottom))

        tvloss += tf.reduce_mean(left_top_diff+left_bottom_diff+right_top_diff+right_bottom_diff)

    return tvloss

# Backbone
class ResBlock(tf.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = Sequential([
            InstanceNormalization(axis=3),
            ReLU(),
            Conv2D(in_channels, kernel_size=3, padding='same', use_bias=False),
            InstanceNormalization(axis=3),
            ReLU(),
            Conv2D(in_channels, kernel_size=3, padding='same', use_bias=False)
        ])

    def call(self, x):
        return self.block(x) + x
    

class DownSample(tf.Module):
    def __init__(self, in_channels, filters):
        super(DownSample, self).__init__()
        self.block = Sequential([
            InstanceNormalization(axis=in_channels),
            #ReLU(),
            Conv2D(filters, kernel_size=3, activation=tf.nn.leaky_relu, strides=2, padding='same', use_bias=False)
        ])

    def call(self, x):
        return self.block(x)


class FeatureEncoder(tf.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = Sequential([
                    DownSample(in_channels, out_chns),
                    ResBlock(out_chns),
                    ResBlock(out_chns)
                ])
            else:
                encoder = Sequential([
                    DownSample(chns[i-1], out_chns),
                    ResBlock(out_chns),
                    ResBlock(out_chns)
                ])
            self.encoders.append(encoder)

        self.encoders = Sequential(self.encoders)

    def call(self, x):
        encoder_features = []
        for encoder in self.encoders.layers:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features
    

class RefinePyramid(tf.Module):
    def __init__(self, chns=[64, 128, 256, 256, 256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = Conv2D(in_chns, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = Sequential(self.adaptive)
        
        # output conv
        self.smooth = []
        for _ in range(len(chns)):
            smooth_layer = Conv2D(fpn_dim, kernel_size=3, padding='same')
            self.smooth.append(smooth_layer)
        self.smooth = Sequential(self.smooth)

    def call(self, x):
        conv_ftr_list = x

        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive(conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + UpSampling2D(size=(2, 2))(last_feature)
            # smooth
            feature = self.smooth(feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))
    
class AFlowNet_Vitonhd_lrarms(tf.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Vitonhd_lrarms, self).__init__()
        self.netLeftMain = []
        self.netTorsoMain = []
        self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            netLeftMain_layer = Sequential(
                Conv2D(in_channels=49, filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netTorsoMain_layer = Sequential(
                Conv2D(in_channels=49, filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRightMain_layer = Sequential(
                Conv2D(in_channels=49, filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netRefine_left_layer = Sequential(
                Conv2D(2 * fpn_dim, filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_torso_layer = Sequential(
                Conv2D(2 * fpn_dim, filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )
            netRefine_right_layer = Sequential(
                Conv2D(input_shape=(None, None, 2 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            )

            netAttentionRefine_layer = Sequential(
                Conv2D(input_shape=(None, None, 4 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=3,
                                kernel_size=3, stride=1, padding=1),
                tf.tanh()
            )

            netSeg_layer = Sequential(
                Conv2D(input_shape=(None, None, fpn_dim*2), filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(filters=64, kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(filters=32, kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(filters=7, kernel_size=3, stride=1, padding=1),
                tf.tanh()
            )

            partFusion_layer = Sequential(
                Conv2D(input_shape=(None, None, fpn_dim*3), kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        # NOTICE
        '''
        self.netLeftMain = Sequential(self.netLeftMain)
        self.netTorsoMain = Sequential(self.netTorsoMain)
        self.netRightMain = Sequential(self.netRightMain)

        self.netLeftRefine = Sequential(self.netLeftRefine)
        self.netTorsoRefine = Sequential(self.netTorsoRefine)
        self.netRightRefine = Sequential(self.netRightRefine)

        self.netAttentionRefine = Sequential(self.netAttentionRefine)
        self.netPartFusion = Sequential(self.netPartFusion)
        self.netSeg = Sequential(self.netSeg)
        '''
        self.softmax = Softmax(dim=1)


    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array_tf = tf.transpose(tf.convert_to_tensor(weight_array, dtype=tf.float32), perm=[3, 2, 0, 1])
        self.weight = tf.Variable(initial_value=weight_array_tf, trainable=False)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_cond_concate = tf.concat([x_cond,x_cond,x_cond],0)
            x_warp_concate = tf.concat([x_warp,x_warp,x_warp],0)

            if last_flow is not None and warp_feature:
                x_warp_after = grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1), padding_mode='border')
            else:
                x_warp_after = x_warp_concate

            tenCorrelation = tf.nn.leaky_relu(correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), alpha=0.1)
            '''
            tenCorrelation = LeakyReLU(input=correlation.FunctionCorrelation(
                                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), negative_slope=0.1, inplace=False)
            '''
            
            bz = x_cond.size(0)

            left_tenCorrelation = tenCorrelation[0:bz]
            torso_tenCorrelation = tenCorrelation[bz:2*bz]
            right_tenCorrelation = tenCorrelation[2*bz:]

            left_flow = self.netLeftMain[i](left_tenCorrelation)
            torso_flow = self.netTorsoMain[i](torso_tenCorrelation)
            right_flow = self.netRightMain[i](right_tenCorrelation)

            flow = tf.concat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = grid_sample(last_flow, flow, padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp_concate = grid_sample(x_warp_concate, flow.permute(
                0, 2, 3, 1), padding_mode='border')

            left_concat = tf.concat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = tf.concat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = tf.concat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            x_attention = tf.concat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)

            left_flow = self.netLeftRefine[i](left_concat)
            torso_flow = self.netTorsoRefine[i](torso_concat)
            right_flow = self.netRightRefine[i](right_concat)   

            flow = tf.concat([left_flow,torso_flow,right_flow],0)
            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                         flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                         flow[2*bz:] * fused_attention[:,2:3,...]
            last_fused_flow = interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_full_warp = grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = tf.split(last_flow, 1, axis=1)
            #flow_x, flow_y = torch.split(last_flow, 1, dim=1)
            delta_x = Conv2D().set_weights([self.weight])
            delta_y = Conv2D().set_weights([self.weight])
            delta_x = delta_x(flow_x)
            delta_y = delta_y(flow_y)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = tf.concat([x_warp,x_warp,x_warp],0)
            x_warp = interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]

            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            x_warp = tf.concat([x_warp_left,x_warp_torso,x_warp_right],1)
            x_warp = self.netPartFusion[i](x_warp)

            concate = tf.concat([x_warp,x_cond],1)
            seg = self.netSeg[i](concate)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
                x_edge_full_all, attention_all, seg_list

class SPADE(tf.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = InstanceNormalization(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        self.mlp_shared = Sequential(
            Conv2D(label_nc, nhidden, kernel_size=3, stride=1, padding=1),
            ReLU()
        )
        self.mlp_gamma = Conv2D(nhidden, norm_nc, kernel_size=3, stride=1, padding=1)
        self.mlp_beta = Conv2D(nhidden, norm_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
    
class ResBlock_SPADE(tf.Module):
    def __init__(self, in_channels):
        super(ResBlock_SPADE, self).__init__()
        self.norm_0 = SPADE(in_channels,1)
        self.norm_1 = SPADE(in_channels,1)

        self.actvn_0 = LeakyReLU(inplace=False, negative_slope=0.1)
        self.actvn_1 = LeakyReLU(inplace=False, negative_slope=0.1)

        self.conv_0 = Conv2D(in_channels,in_channels,kernel_size=3, padding=1)
        self.conv_1 = Conv2D(in_channels,in_channels,kernel_size=3, padding=1)

    def forward(self, x, label_map):
        dx = self.conv_0(self.actvn_0(self.norm_0(x, label_map)))
        dx = self.conv_1(self.actvn_1(self.norm_1(dx, label_map)))

        return dx + x


class FeatureEncoder_SPADE(tf.Module):
    def __init__(self, in_channels, chns=[64, 128, 256, 256, 256]):
        super(FeatureEncoder_SPADE, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = Sequential([DownSample(in_channels, out_chns),
                                        ResBlock_SPADE(out_chns),
                                        ResBlock_SPADE(out_chns)])
            else:
                encoder = Sequential([DownSample(chns[i-1], out_chns),
                                        ResBlock_SPADE(out_chns),
                                        ResBlock_SPADE(out_chns)])

            self.encoders.append(encoder)

        self.encoders = Sequential(self.encoders)

    def forward(self, x, label_map):
        encoder_features = []
        for encoder in self.encoders:
            for ii, encoder_submodule in enumerate(encoder):
                if ii == 0:
                    x = encoder_submodule(x)
                else:
                    x = encoder_submodule(x, label_map)
            encoder_features.append(x)
        return encoder_features

class AFlowNet_Dresscode_lrarms(tf.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet_Dresscode_lrarms, self).__init__()
        self.netLeftMain = []
        self.netTorsoMain = []
        self.netRightMain = []

        self.netLeftRefine = []
        self.netTorsoRefine = []
        self.netRightRefine = []

        self.netAttentionRefine = []
        self.netPartFusion = []
        self.netSeg = []

        for i in range(num_pyramid):
            netLeftMain_layer = Sequential([
                Conv2D(input_shape=(None, None, 49), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netTorsoMain_layer = Sequential([
                Conv2D(input_shape=(None, None, 49), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRightMain_layer = Sequential([
                Conv2D(input_shape=(None, None, 49), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])

            netRefine_left_layer = Sequential([
                Conv2D(input_shape=(None, None, 2 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRefine_torso_layer = Sequential([
                Conv2D(input_shape=(None, None, 2 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])
            netRefine_right_layer = Sequential([
                Conv2D(input_shape=(None, None, 2 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=2,
                                kernel_size=3, stride=1, padding=1)
            ])

            netAttentionRefine_layer = Sequential(
                Conv2D(input_shape=(None, None, 4 * fpn_dim), filters=128,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=3,
                                kernel_size=3, stride=1, padding=1),
                tf.tanh()
            )

            netSeg_layer = Sequential([
                Conv2D(input_shape=(None, None, fpn_dim*2), filters=128,
                                kernel_size=3, stride=1, padding=1),
                SPADE(128,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=128, filters=64,
                                kernel_size=3, stride=1, padding=1),
                SPADE(64,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=64, filters=32,
                                kernel_size=3, stride=1, padding=1),
                SPADE(32,1),
                LeakyReLU(inplace=False, negative_slope=0.1),
                Conv2D(in_channels=32, filters=10,
                                kernel_size=3, stride=1, padding=1),
                tf.tanh()
            ])

            partFusion_layer = Sequential(
                Conv2D(input_shape=(None, None, fpn_dim*3), filters=fpn_dim, kernel_size=1),
                ResBlock(fpn_dim)
            )

            self.netLeftMain.append(netLeftMain_layer)
            self.netTorsoMain.append(netTorsoMain_layer)
            self.netRightMain.append(netRightMain_layer)

            self.netLeftRefine.append(netRefine_left_layer)
            self.netTorsoRefine.append(netRefine_torso_layer)
            self.netRightRefine.append(netRefine_right_layer)

            self.netAttentionRefine.append(netAttentionRefine_layer)
            self.netPartFusion.append(partFusion_layer)
            self.netSeg.append(netSeg_layer)

        '''
        self.netLeftMain = Sequential(self.netLeftMain)
        self.netTorsoMain = Sequential(self.netTorsoMain)
        self.netRightMain = Sequential(self.netRightMain)

        self.netLeftRefine = Sequential(self.netLeftRefine)
        self.netTorsoRefine = Sequential(self.netTorsoRefine)
        self.netRightRefine = Sequential(self.netRightRefine)

        self.netAttentionRefine = Sequential(self.netAttentionRefine)
        self.netPartFusion = Sequential(self.netPartFusion)
        self.netSeg = Sequential(self.netSeg)
        '''
        self.softmax = Softmax(dim=1)


    def forward(self, x, x_edge, x_full, x_edge_full, x_warps, x_conds, preserve_mask, cloth_label_map, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        x_full_all = []
        x_edge_full_all = []
        attention_all = []
        seg_list = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        
        

        weight_array = tf.transpose(tf.convert_to_tensor(weight_array, dtype=tf.float32), perm=[3, 2, 0, 1])
        
        self.weight = tf.Variable(initial_value=weight_array, trainable=False)

        for i in range(len(x_warps)):
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_cond_concate = tf.concat([x_cond,x_cond,x_cond],0)
            x_warp_concate = tf.concat([x_warp,x_warp,x_warp],0)

            if last_flow is not None and warp_feature:
                x_warp_after = grid_sample(x_warp_concate, last_flow.detach().permute(0, 2, 3, 1),
                                             mode='bilinear', padding_mode='border')
            else:
                x_warp_after = x_warp_concate

            tenCorrelation = LeakyReLU(input=correlation.FunctionCorrelation(
                tenFirst=x_warp_after, tenSecond=x_cond_concate, intStride=1), alpha=0.1)
            
            bz = x_cond.size(0)

            left_flow = tenCorrelation[0:bz]
            torso_flow = tenCorrelation[bz:2*bz]
            right_flow = tenCorrelation[2*bz:]

            for ii, sub_flow_module in enumerate(self.netLeftMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    left_flow = sub_flow_module(left_flow, cloth_label_map)
                else:
                    left_flow = sub_flow_module(left_flow)

            for ii, sub_flow_module in enumerate(self.netTorsoMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    torso_flow = sub_flow_module(torso_flow, cloth_label_map)
                else:
                    torso_flow = sub_flow_module(torso_flow)

            for ii, sub_flow_module in enumerate(self.netRightMain[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    right_flow = sub_flow_module(right_flow, cloth_label_map)
                else:
                    right_flow = sub_flow_module(right_flow)

            flow = tf.concat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            if last_flow is not None:
                flow = grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
            else:
                flow = flow.permute(0, 3, 1, 2)

            last_flow = flow
            x_warp_concate = grid_sample(x_warp_concate, flow.permute(
                0, 2, 3, 1), mode='bilinear', padding_mode='border')

            left_concat = tf.concat([x_warp_concate[0:bz], x_cond_concate[0:bz]], 1)
            torso_concat = tf.concat([x_warp_concate[bz:2*bz], x_cond_concate[bz:2*bz]],1)
            right_concat = tf.concat([x_warp_concate[2*bz:], x_cond_concate[2*bz:]],1)

            x_attention = tf.concat([x_warp_concate[0:bz],x_warp_concate[bz:2*bz],x_warp_concate[2*bz:],x_cond],1)
            fused_attention = self.netAttentionRefine[i](x_attention)
            fused_attention = self.softmax(fused_attention)

            left_flow = left_concat
            torso_flow = torso_concat
            right_flow = right_concat

            for ii, sub_flow_module in enumerate(self.netLeftRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    left_flow = sub_flow_module(left_flow, cloth_label_map)
                else:
                    left_flow = sub_flow_module(left_flow)

            for ii, sub_flow_module in enumerate(self.netTorsoRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    torso_flow = sub_flow_module(torso_flow, cloth_label_map)
                else:
                    torso_flow = sub_flow_module(torso_flow)

            for ii, sub_flow_module in enumerate(self.netRightRefine[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    right_flow = sub_flow_module(right_flow, cloth_label_map)
                else:
                    right_flow = sub_flow_module(right_flow)     

            flow = tf.concat([left_flow,torso_flow,right_flow],0)

            delta_list.append(flow)
            flow = apply_offset(flow)
            flow = grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

            fused_flow = flow[0:bz] * fused_attention[:,0:1,...] + \
                         flow[bz:2*bz] * fused_attention[:,1:2,...] + \
                         flow[2*bz:] * fused_attention[:,2:3,...]
            last_fused_flow = interpolate(fused_flow, scale_factor=2, mode='bilinear')

            fused_attention = interpolate(fused_attention, scale_factor=2, mode='bilinear')
            attention_all.append(fused_attention)

            cur_x_full = interpolate(x_full, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_full_warp = grid_sample(cur_x_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_full_all.append(cur_x_full_warp)
            cur_x_edge_full = interpolate(x_edge_full, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_edge_full_warp = grid_sample(cur_x_edge_full, last_fused_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_full_all.append(cur_x_edge_full_warp)

            last_flow = interpolate(flow, scale_factor=2, mode='bilinear')
            last_flow_all.append(last_flow)

            cur_x = interpolate(x, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            cur_x_warp = grid_sample(cur_x, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_all.append(cur_x_warp)
            cur_x_edge = interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
            cur_x_warp_edge = grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')
            x_edge_all.append(cur_x_warp_edge)

            flow_x, flow_y = tf.split(last_flow, 1, dim=1)
            delta_x = Conv2D().set_weights([self.weight])
            delta_x = delta_x(flow_x)
            delta_y = Conv2D().set_weights([self.weight])
            delta_y = delta_y(flow_y)
            delta_x_all.append(delta_x)
            delta_y_all.append(delta_y)

            # predict seg
            cur_preserve_mask = interpolate(preserve_mask, scale_factor=0.5 ** (len(x_warps)-1-i), mode='bilinear')
            x_warp = x_warps[len(x_warps) - 1 - i]
            x_cond = x_conds[len(x_warps) - 1 - i]

            x_warp = tf.concat([x_warp,x_warp,x_warp],0)
            x_warp = interpolate(x_warp, scale_factor=2, mode='bilinear')
            x_cond = interpolate(x_cond, scale_factor=2, mode='bilinear')

            x_warp = grid_sample(x_warp, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
            x_warp_left = x_warp[0:bz]
            x_warp_torso = x_warp[bz:2*bz]
            x_warp_right = x_warp[2*bz:]

            x_edge_left = cur_x_warp_edge[0:bz]
            x_edge_torso = cur_x_warp_edge[bz:2*bz]
            x_edge_right = cur_x_warp_edge[2*bz:]

            x_warp_left = x_warp_left * x_edge_left * (1-cur_preserve_mask)
            x_warp_torso = x_warp_torso * x_edge_torso * (1-cur_preserve_mask)
            x_warp_right = x_warp_right * x_edge_right * (1-cur_preserve_mask)

            x_warp = tf.concat([x_warp_left,x_warp_torso,x_warp_right],1)
            x_warp = self.netPartFusion[i](x_warp)

            seg = tf.concat([x_warp, x_cond],1)
            for ii, sub_flow_module in enumerate(self.netSeg[i]):
                if ii == 1 or ii == 4 or ii == 7:
                    seg = sub_flow_module(seg, cloth_label_map)
                else:
                    seg = sub_flow_module(seg)
            seg_list.append(seg)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, x_full_all, \
                x_edge_full_all, attention_all, seg_list


class AFWM_Dressescode_lrarms(tf.Module):
    def __init__(self, opt, input_nc, clothes_input_nc=3):
        super(AFWM_Dressescode_lrarms, self).__init__()
        num_filters = [64, 128, 256, 256, 256]
        # num_filters = [64,128,256,512,512]
        fpn_dim = 256
        self.image_features = FeatureEncoder_SPADE(clothes_input_nc+1, num_filters)
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        self.cond_FPN = RefinePyramid(chns=num_filters, fpn_dim=fpn_dim)
        
        self.aflow_net = AFlowNet_Dresscode_lrarms(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge, image_label_input, image_input_left, image_input_torso, \
                image_input_right, image_edge_left, image_edge_torso, image_edge_right, preserve_mask, cloth_label_map):
        image_input_concat = tf.concat([image_input, image_label_input],1)

        image_pyramids = self.image_FPN(self.image_features(image_input_concat, cloth_label_map))
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input))  # maybe use nn.Sequential

        image_concat = tf.concat([image_input_left,image_input_torso,image_input_right],0)
        image_edge_concat = tf.concat([image_edge_left, image_edge_torso, image_edge_right],0)

        last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
            x_full_all, x_edge_full_all, attention_all, seg_list = self.aflow_net(image_concat, \
            image_edge_concat, image_input, image_edge, image_pyramids, cond_pyramids, \
            preserve_mask, cloth_label_map)

        return last_flow, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all, \
                x_full_all, x_edge_full_all, attention_all, seg_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr