import pywt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
import cv2
import pywt
import numpy as np
import skimage.feature as feature
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from skimage.restoration import denoise_nl_means
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
import torch
import torch.nn.functional as F

import utils.MultiFeaturesExtractorSAR as sar_extractor
import utils.MultiFeaturesExtractorMon as mon_extractor
import utils.MultiFeaturesExtractorRD as rd_extractor

feature_name = [
    # 统计域 : rms
    # 时频域：wavelet
    # 空域：Gabor 纹理特征 、 HOG
]


def get_statis_feat(data, k=7, s=4, p=3):
    data = cv2.resize(data, (32, 32))
    h, w = data.shape

    data = np.pad(data, pad_width=p, mode='constant', constant_values=0)

    # 计算输出矩阵的形状
    out_h = (h - k + 2 * p) // s + 1
    out_w = (w - k + 2 * p) // s + 1

    rms_matrix = np.zeros((out_h, out_w))

    from numpy.lib.stride_tricks import as_strided

    new_shape = (out_h, out_w, k, k)
    new_strides = (data.strides[0] * s, data.strides[1] * s, data.strides[0], data.strides[1])

    windows = as_strided(data, shape=new_shape, strides=new_strides)
    rms_matrix = np.sqrt(np.mean(np.square(windows), axis=(2, 3))).flatten()
    std_matrix = np.std(windows, axis=(2, 3)).flatten()
    max_matrix = np.max(windows, axis=(2, 3)).flatten()
    mean_matrix = np.mean(windows, axis=(2, 3)).flatten()

    statis_feat = np.stack([rms_matrix, std_matrix, max_matrix, mean_matrix])
    return statis_feat


def get_time_freq_feat(data, level=2):
    data = cv2.resize(data, (32, 32))
    coeffs = pywt.wavedec2(data, 'haar', level=level)
    ll = coeffs[0]
    high_lv_bands = [c for c in coeffs[1]]  # 只获取最高级别频带
    high_lv_bands.extend([ll])

    time_freq_features = np.array(high_lv_bands).reshape(4, -1)
    return time_freq_features


def get_spatial_feat(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''
    data = cv2.resize(data, (32, 32))

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    data = np.log1p(data).astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_outra_feature(data, cfg):
    """
    安全地从配置中提取特征
    
    Args:
        data: 输入数据
        cfg: 配置字典，包含 'feature_combinations' 键
    
    Returns:
        特征列表
    """
    feat = []
    H, W = data.shape
    for fun_name in cfg['feature_combinations']:
        # 从当前模块的全局命名空间中查找函数
        func = globals().get(fun_name)

        # 检查函数是否存在
        if func is not None:
            data = data.reshape(H, W)
        else:
            # 调用外部优选特征
            mode = cfg['data']['mode']
            assert mode in ['RD', 'SAR', 'MON']
            if mode == 'SAR':
                func = getattr(sar_extractor, fun_name, None)
                data = data.reshape(1, H, W, 1)
            elif mode == 'RD':
                func = getattr(rd_extractor, fun_name, None)
                data = data.reshape(1, 1, H, W, 1)
            elif mode == 'MON':
                func = getattr(mon_extractor, fun_name, None)
                data = data.reshape(1, H, W, 1)
            if func is None:
                print(f"Warning: Function '{fun_name}' not found in current module. Skipping.")
                continue

        # 检查是否可调用
        if not callable(func):
            print(f"Warning: '{fun_name}' is not callable. Skipping.")
            continue

        try:
            result = func(data)
            if not isinstance(result, np.ndarray):
                result = np.array(result)

            result_flat = result.flatten()
            feat.extend(result_flat.tolist()[:256])  # 只取前256个特征

        except Exception as e:
            print(f"Error calling function '{fun_name}': {str(e)}")
            continue
    feat = np.array(feat).astype(np.float32)

    return feat


if __name__ == '__main__':

    data = np.random.rand(32, 32)  # 输入数据归一化0-1

    hog_features = get_spatial_feat(data)
    print(hog_features.shape)  # (9, 64)
    # print(hog_features)

    statis_feat = get_statis_feat(data)
    print(statis_feat.shape)  # (2, 64)
    # print(hog_features)

    time_freq_features = get_time_freq_feat(data)
    print(time_freq_features.shape)  # (4, 64)


def get_test_feat1(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat2(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat3(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat4(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features
