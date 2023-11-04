import os

import numpy as np
import scipy.ndimage as ndimage
import SimpleITK as sitk


def preprocess_nii(img_path):
    # read nifit format, set window，reset 0 to 255
    sitkImage = sitk.ReadImage(img_path)
    intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
    # window choose
    intensityWindowingFilter.SetWindowMaximum(400)
    intensityWindowingFilter.SetWindowMinimum(15)
    # 0 to 255
    intensityWindowingFilter.SetOutputMaximum(255)
    intensityWindowingFilter.SetOutputMinimum(0)
    # resample size
    sitkImage = resampleSize(sitkImage, 32)
    # filter
    sitkImage = intensityWindowingFilter.Execute(sitkImage)
    return sitkImage

def resampleSize(sitkImage, depth):
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing/(depth/float(zsize))
    # new_spacing_x = xspacing/(256/float(xsize))
    # new_spacing_y = yspacing/(256/float(ysize))
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #based on new spacing calculate new size
    newsize = (xsize, ysize, depth)
    newspace = (xspacing, yspacing, new_spacing_z)
    # newsize = (256, 256, depth)
    # newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage


def exact_skull(path):
    """
    It is not the complete CT preprocessment, just the part in Python.
    """
    image = preprocess_nii(path)
    data = sitk.GetArrayFromImage(image)

    threshold = 50
    binary_image = data < threshold

    binary_image = ndimage.binary_opening(binary_image)
    label_image, num_labels = ndimage.label(binary_image)

    largest_label = np.argmax(np.bincount(label_image.flat)[1:]) + 1

    brain_mask = label_image == largest_label

    skull_stripped_data = data * brain_mask
    # Normalization
    for i in range(32):
        for j in range(512):
            for k in range(512):
                skull_stripped_data[i][j][k]=(skull_stripped_data[i][j][k]-2.4908)/6.2173
    skull_stripped_image = sitk.GetImageFromArray(skull_stripped_data)
    skull_stripped_image.CopyInformation(image)
    sitk.WriteImage(skull_stripped_image, path)


def get_mean_and_std(good_path, bad_path):
    good = os.listdir(good_path)
    bad = os.listdir(bad_path)
    good_len = len(good)
    bad_len = len(bad)
    good_img_list = []
    for item in good:
        img = sitk.ReadImage(good_path + '/' + item)
        img_narry = sitk.GetArrayFromImage(img)
        good_img_list = np.concatenate((good_img_list, img_narry.flatten()), axis=0)
        print(item)
    good_mean = np.mean(good_img_list)
    good_std = np.std(good_img_list)

    bad_img_list = []
    for item in bad:
        img = sitk.ReadImage(bad_path + '/' + item)
        img_narry = sitk.GetArrayFromImage(img)
        bad_img_list = np.concatenate((bad_img_list, img_narry.flatten()), axis=0)
        print(item)
    bad_mean = np.mean(bad_img_list)
    bad_std = np.std(bad_img_list)

    mean = (good_mean * good_len + bad_mean * bad_len) / (good_len + bad_len)
    a = (good_len - 1) * (good_std ** 2) + (bad_len - 1) * (bad_std ** 2) + good_len * bad_len / (
                good_len + bad_len) * (good_mean ** 2 + bad_mean ** 2 - 2 * good_mean * bad_mean)
    std = (a / (good_len + bad_len - 1)) ** 0.5

    return mean,std



