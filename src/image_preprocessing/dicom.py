import warnings

warnings.filterwarnings("ignore")

import os
import dicomsdl
import numpy as np
import pandas as pd
import pydicom
import torch
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut, pixel_dtype
from tqdm import tqdm

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

from src.image_preprocessing import misc as misc_utils
from torch.nn import functional as F
from tqdm import tqdm

from src.image_preprocessing.windowing import apply_windowing

TORCH_DTYPES = {
    'uint8': torch.uint8,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


class PydicomMetadata:
    """
    A class to extract and store essential DICOM metadata related to 
    windowing and display interpretation.

    Attributes:
        window_widths (list): A list of window width values (float).
        window_centers (list): A list of window center values (float).
        voilut_func (str): The VOI LUT function (default: "LINEAR").
        invert (bool): True if the image uses MONOCHROME1 (inverted display mode).
    """

    def __init__(self, ds):
        if "WindowWidth" not in ds or "WindowCenter" not in ds:
            self.window_widths = []
            self.window_centers = []
        else:
            ww = ds['WindowWidth']
            wc = ds['WindowCenter']
            self.window_widths = [float(e) for e in ww
                                  ] if ww.VM > 1 else [float(ww.value)]

            self.window_centers = [float(e) for e in wc
                                   ] if wc.VM > 1 else [float(wc.value)]

        # if nan --> LINEAR
        self.voilut_func = str(ds.get('VOILUTFunction', 'LINEAR')).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)


class DicomsdlMetadata:
    """
    A class to extract and store essential DICOM metadata using the dicomsdl library.

    Attributes:
        window_widths (list): A list of window width values (float).
        window_centers (list): A list of window center values (float).
        voilut_func (str): The VOI LUT function (default: "LINEAR").
        invert (bool): True if the image uses MONOCHROME1 (inverted display mode).
    """

    def __init__(self, ds):
        self.window_widths = ds.WindowWidth
        self.window_centers = ds.WindowCenter
        if self.window_widths is None or self.window_centers is None:
            self.window_widths = []
            self.window_centers = []
        else:
            try:
                if not isinstance(self.window_widths, list):
                    self.window_widths = [self.window_widths]
                self.window_widths = [float(e) for e in self.window_widths]
                if not isinstance(self.window_centers, list):
                    self.window_centers = [self.window_centers]
                self.window_centers = [float(e) for e in self.window_centers]
            except:
                self.window_widths = []
                self.window_centers = []

        # if nan --> LINEAR
        self.voilut_func = ds.VOILUTFunction
        if self.voilut_func is None:
            self.voilut_func = 'LINEAR'
        else:
            self.voilut_func = str(self.voilut_func).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)


def min_max_scale(img):
    maxv = img.max()
    minv = img.min()
    if maxv > minv:
        return (img - minv) / (maxv - minv)
    else:
        return img - minv  # ==0


#@TODO: percentile on both min-max ?
# this version is not correctly implemented, but used in the winning submission
def percentile_min_max_scale(img, pct=99):
    if isinstance(img, np.ndarray):
        maxv = np.percentile(img, pct) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = np.clip(ret, 0, 1)
    elif isinstance(img, torch.Tensor):
        maxv = torch.quantile(img, pct / 100) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = torch.clamp(ret, 0, 1)
    else:
        raise ValueError(
            'Invalid img type, should be numpy array or torch.Tensor')
    return ret


# @TODO: support windowing with more bits (>8)
def normalize_dicom_img(img,
                        invert,
                        save_dtype,
                        window_centers,
                        window_widths,
                        window_func,
                        window_index=0,
                        method='windowing',
                        force_use_gpu=True):
    assert method in ['min_max', 'min_max_pct', 'windowing']
    assert save_dtype in ['uint8', 'uint16', 'float16', 'float32', 'float64']
    if save_dtype == 'uint16':
        if invert:
            img = img.max() - img
        return img

    if method == 'windowing':
        assert save_dtype == 'uint8', 'Currently `windowing` normalization only support `uint8` save dtype.'
        # apply windowing
        if len(window_centers) > 0:
            window_center = window_centers[window_index]
            window_width = window_widths[window_index]
            windowing_backend = 'torch' if isinstance(
                img, torch.Tensor) or force_use_gpu else 'np_v2'
            img = apply_windowing(img,
                                  window_width=window_width,
                                  window_center=window_center,
                                  voi_func=window_func,
                                  y_min=0,
                                  y_max=255,
                                  backend=windowing_backend)
        # if no window center/width in dcm file
        # do simple min-max scaling
        else:
            print(
                'No windowing param, perform min-max scaling normalization instead.'
            )
            img = min_max_scale(img)
            img = img * 255
        img = img.to(torch.uint8)
        return img
    elif method == 'min_max':
        # [0, 1]
        img = min_max_scale(img)
    elif method == 'min_max_pct':
        # [0, 1]
        img = percentile_min_max_scale(img)
    else:
        raise ValueError(f'Invalid normalization method `{method}`')
    if invert:
        img = 1.0 - img
    if save_dtype == 'uint8':
        img = img * 255
    # convert to specified dtype: uint8, float
    if isinstance(img, np.ndarray):
        img = img.astype(save_dtype)
    elif isinstance(img, torch.Tensor):
        img = img.to(TORCH_DTYPES[save_dtype])

    return img


#################################### DICOMSDL ####################################
def _convert_single_with_dicomsdl(dcm_path,
                                  save_path,
                                  normalization='windowing',
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  index=0,
                                  legacy=False):
    dcm = dicomsdl.open(dcm_path)
    meta = DicomsdlMetadata(dcm)
    info = dcm.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]

    ori_dtype = info['dtype']
    img = np.empty(shape, dtype=ori_dtype)
    dcm.copyFrameData(index, img)

    # legacy: old method (cpu numpy operation), for compatibility only
    # new method: gpu torch operation to improve speed
    if not legacy:
        img = torch.from_numpy(img.astype(np.int16)).cuda()
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_dicomsdl(dcm_paths,
                          save_paths,
                          normalization='windowing',
                          save_backend='cv2',
                          save_dtype='uint8',
                          legacy=False):
    assert len(dcm_paths) == len(save_paths)
    for i in tqdm(range(len(dcm_paths))):
        _convert_single_with_dicomsdl(dcm_paths[i],
                                      save_paths[i],
                                      normalization=normalization,
                                      save_backend=save_backend,
                                      save_dtype=save_dtype,
                                      legacy=legacy)
    return


def convert_with_dicomsdl_parallel(dcm_paths,
                                   save_paths,
                                   normalization='windowing',
                                   save_backend='cv2',
                                   save_dtype='uint8',
                                   parallel_n_jobs=2,
                                   joblib_backend='loky',
                                   legacy=False):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dicomsdl(dcm_paths,
                                     save_paths,
                                     normalization=normalization,
                                     save_backend=save_backend,
                                     save_dtype=save_dtype,
                                     legacy=legacy)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`')
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(_convert_single_with_dicomsdl)(dcm_paths[j],
                                                   save_paths[j],
                                                   normalization=normalization,
                                                   save_backend=save_backend,
                                                   save_dtype=save_dtype,
                                                   legacy=legacy)
            for j in tqdm(range(len(dcm_paths))))


def _convert_single_with_pydicom(dcm_path,
                                 save_path,
                                 normalization='windowing',
                                 save_backend='cv2',
                                 save_dtype='uint8'):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array
    meta = PydicomMetadata(ds)
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_pydicom(dcm_paths,
                         save_paths,
                         normalization='windowing',
                         save_backend='cv2',
                         save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    for dcm_path, save_path in tqdm(zip(dcm_paths, save_paths)):
        _convert_single_with_pydicom(dcm_path, save_path, normalization,
                                     save_backend, save_dtype)


def convert_with_pydicom_parallel(dcm_paths,
                                  save_paths,
                                  normalization='windowing',
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  parallel_n_jobs=2,
                                  parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_pydicom(dcm_paths, save_paths, normalization,
                                    save_backend, save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(_convert_single_with_pydicom)(
                dcm_paths[j], save_paths[j], normalization, save_backend,
                save_dtype) for j in tqdm(range(len(dcm_paths))))
