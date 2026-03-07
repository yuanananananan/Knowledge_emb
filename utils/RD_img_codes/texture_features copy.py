import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from joblib import Parallel, delayed


# ======================
# 顶层 GLCM helpers
# ======================
def _glcm_feature(img, prop, distances, angles, levels):
    glcm = graycomatrix(img.astype(np.uint8),
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)
    val = graycoprops(glcm, prop).astype(np.float32).mean()

    return np.float16(val)

def glcm_contrast(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_glcm_feature)(im, 'contrast', distances, angles, levels) for im in images
    ), dtype=np.float16)

def glcm_dissimilarity(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_glcm_feature)(im, 'dissimilarity', distances, angles, levels) for im in images
    ), dtype=np.float16)

def glcm_homogeneity(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_glcm_feature)(im, 'homogeneity', distances, angles, levels) for im in images
    ), dtype=np.float16)

def glcm_correlation(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_glcm_feature)(im, 'correlation', distances, angles, levels) for im in images
    ), dtype=np.float16)

def glcm_energy(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_glcm_feature)(im, 'energy', distances, angles, levels) for im in images
    ), dtype=np.float16)


# ======================
# 顶层 LBP helpers
# ======================
def _lbp_hist(img, P, R, method, n_bins, normalize):
    lbp = local_binary_pattern(img.astype(np.uint8), P, R, method).astype(np.float32)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins+1), density=normalize)
    return hist.astype(np.float16)

def lbp_histogram(images, P=4, R=1, method='default', normalize=True, n_jobs=-1):
    n_bins = P + 2 if method == 'uniform' else 2**P
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_lbp_hist)(im, P, R, method, n_bins, normalize) for im in images
    ), dtype=np.float16)


# ======================
# 顶层 Gabor helpers
# ======================
# def _gabor_magnitude(img, frequency, theta):
#     real, imag = gabor(img, frequency=frequency, theta=theta)
#     return np.sqrt(real**2 + imag**2).astype(np.float16)

def _gabor_magnitude(img, frequency, theta):
    img = img.astype(np.float32)  # 输入转 float32
    real, imag = gabor(img, frequency=frequency, theta=theta)
    # 中间数组转 float32
    mag = np.sqrt(real.astype(np.float32)**2 + imag.astype(np.float32)**2)
    return mag

def _gabor_mean(img, frequency, theta):
    return np.float16(_gabor_magnitude(img, frequency, theta).mean())

def _gabor_variance(img, frequency, theta):
    return np.float16(_gabor_magnitude(img, frequency, theta).var())

def _gabor_energy(img, frequency, theta):
    mag = _gabor_magnitude(img, frequency, theta)
    return np.float16(np.mean(mag**2))

def gabor_mean(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_gabor_mean)(im, frequency, theta) for im in images
    ), dtype=np.float16)

def gabor_variance(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_gabor_variance)(im, frequency, theta) for im in images
    ), dtype=np.float16)

def gabor_energy(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_gabor_energy)(im, frequency, theta) for im in images
    ), dtype=np.float16)
