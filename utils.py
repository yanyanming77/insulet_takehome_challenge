# utility files
# defined functions, assets saved from training process

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pywt
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# function: create date variables
def create_date_variables(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # cyclical encoding for month and day of week
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
    df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
    return df

# function: read images
def load_images(df, target_size = (224, 224)):
    imgs_list = []
    for img_path in tqdm(df['image']):
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img = np.array(img, dtype=np.float32) 
            imgs_list.append(img)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            imgs_list.append(np.zeros(target_size[0] * target_size[1] * 3))
    return np.array(imgs_list)

# function: manually extract 25 low-level image features
def extract_minimal_image_features(images):
    """
    Extract 25 low-level features from each image:
    color (mean/std/skew/kurtosis), brightness, contrast, edges,
    texture (entropy, LBP, GLCM), sharpness, saturation, frequency.
    """

    n_samples = images.shape[0]
    features_list = []

    feature_names = [
        # color stats (12)
        'r_mean', 'g_mean', 'b_mean',
        'r_std', 'g_std', 'b_std',
        'r_skew', 'g_skew', 'b_skew',
        'r_kurtosis', 'g_kurtosis', 'b_kurtosis',
        # brightness / contrast (3)
        'brightness_mean', 'brightness_std', 'dynamic_range',
        # edges / sharpness (3)
        'edge_density', 'laplacian_var', 'corner_density',
        # texture (6)
        'entropy', 'lbp_entropy',
        'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity',
        # frequency / saturation (1 + 2)
        'wavelet_energy', 'saturation_mean', 'saturation_std'
    ]

    for i in range(n_samples):
        img = images[i]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_uint8 = cv2.convertScaleAbs(gray)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        feature_vector = []

        # 1. Color stats (mean, std, skew, kurtosis per channel)
        for c in range(3):
            vals = img[:, :, c].flatten().astype(np.float32)
            feature_vector.extend([
                np.mean(vals),
                np.std(vals),
                skew(vals),
                kurtosis(vals)
            ])

        # 2. Brightness & contrast
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        dynamic_range = gray.max() - gray.min()
        feature_vector.extend([brightness_mean, brightness_std, dynamic_range])

        # 3. Edge & sharpness
        gray_uint8 = cv2.convertScaleAbs(gray)
        edges = cv2.Canny(gray_uint8, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        laplacian_var = cv2.Laplacian(gray_uint8, cv2.CV_64F).var()
        corners = cv2.goodFeaturesToTrack(gray_uint8, 100, 0.01, 10)
        corner_density = len(corners) / gray.size if corners is not None else 0
        feature_vector.extend([edge_density, laplacian_var, corner_density])

        # 4. Texture features
        entropy = shannon_entropy(gray)
        lbp = local_binary_pattern(gray, P=8, R=1)
        lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 32), density=True)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))

        glcm = graycomatrix(gray_uint8, distances=[5], angles=[0], symmetric=True, normed=True)
        glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
        glcm_corr = graycoprops(glcm, 'correlation')[0, 0]
        glcm_energy = graycoprops(glcm, 'energy')[0, 0]
        glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        feature_vector.extend([
            entropy, lbp_entropy,
            glcm_contrast, glcm_corr, glcm_energy, glcm_homogeneity
        ])

        # 5. Frequency & saturation
        coeffs2 = pywt.dwt2(gray, 'db1')
        cA, (cH, cV, cD) = coeffs2
        wavelet_energy = np.mean(np.square(cA))

        saturation_mean = np.mean(hsv[:, :, 1])
        saturation_std = np.std(hsv[:, :, 1])

        feature_vector.extend([wavelet_energy, saturation_mean, saturation_std])

        features_list.append(np.nan_to_num(feature_vector))

    return np.array(features_list), feature_names