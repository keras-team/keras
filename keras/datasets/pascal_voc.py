#!/usr/bin/env python
# coding=utf-8
"""
This is a script for downloading and converting the pascal voc 2012 dataset
and the berkely extended version.

    # original PASCAL VOC 2012
    # wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # 2 GB

    # berkeley augmented Pascal VOC
    # wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz # 1.3 GB

This can be run as an independent executable to download
the dataset or be imported by scripts used for larger experiments.

If you aren't sure run this to do a full download + conversion setup of the dataset:
   ./data_pascal_voc.py pascal_voc_setup
"""
from __future__ import division, print_function, unicode_literals
import sys
import os
import shutil
import errno
import tarfile
from sacred import Ingredient, Experiment
import numpy as np
from PIL import Image
from collections import defaultdict
from keras.utils import get_file
import skimage.io as io
import numpy as np


def mkdir_p(path):
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def pascal_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.

    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

    enumerated_array = enumerate(class_names[:-1])

    classes_lut = list(enumerated_array)

    # Add a special class representing ambigious regions
    # which has index 255.
    classes_lut.append((255, class_names[-1]))

    classes_lut = dict(classes_lut)

    return classes_lut


def get_pascal_segmentation_images_lists_txts(pascal_root):
    """Return full paths to files in PASCAL VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.

    Parameters
    ----------
    pascal_root : string
        Full path to the root of PASCAL VOC dataset.

    Returns
    -------
    full_filenames_txts : [string, string, string]
        Array that contains paths for train/val/trainval txts with images names.
    """

    segmentation_images_lists_relative_folder = 'ImageSets/Segmentation'

    segmentation_images_lists_folder = os.path.join(pascal_root,
                                                    segmentation_images_lists_relative_folder)

    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')

    pascal_trainval_list_filname = os.path.join(segmentation_images_lists_folder,
                                                'trainval.txt')

    return [
        pascal_train_list_filename,
        pascal_validation_list_filename,
        pascal_trainval_list_filname
    ]


def readlines_with_strip(filename):
    """Reads lines from specified file with whitespaced removed on both sides.
    The function reads each line in the specified file and applies string.strip()
    function to each line which results in removing all whitespaces on both ends
    of each string. Also removes the newline symbol which is usually present
    after the lines wre read using readlines() function.

    Parameters
    ----------
    filename : string
        Full path to the root of PASCAL VOC dataset.

    Returns
    -------
    clean_lines : array of strings
        Strings that were read from the file and cleaned up.
    """

    # Get raw filnames from the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Clean filenames from whitespaces and newline symbols
    clean_lines = map(lambda x: x.strip(), lines)

    return clean_lines


def readlines_with_strip_array_version(filenames_array):
    """The function that is similar to readlines_with_strip() but for array of filenames.
    Takes array of filenames as an input and applies readlines_with_strip() to each element.

    Parameters
    ----------
    array of filenams : array of strings
        Array of strings. Each specifies a path to a file.

    Returns
    -------
    clean_lines : array of (array of strings)
        Strings that were read from the file and cleaned up.
    """

    multiple_files_clean_lines = map(readlines_with_strip, filenames_array)

    return multiple_files_clean_lines


def add_full_path_and_extention_to_filenames(filenames_array, full_path, extention):
    """Concatenates full path to the left of the image and file extention to the right.
    The function accepts array of filenames without fullpath and extention like 'cat'
    and adds specified full path and extetion to each of the filenames in the array like
    'full/path/to/somewhere/cat.jpg.
    Parameters
    ----------
    filenames_array : array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of strings
        updated array with filenames
    """
    full_filenames = map(lambda x: os.path.join(
        full_path, x) + '.' + extention, filenames_array)

    return full_filenames


def add_full_path_and_extention_to_filenames_array_version(filenames_array_array, full_path, extention):
    """Array version of the add_full_path_and_extention_to_filenames() function.
    Applies add_full_path_and_extention_to_filenames() to each element of array.
    Parameters
    ----------
    filenames_array_array : array of array of strings
        Array of strings representing filenames
    full_path : string
        Full path string to be added on the left to each filename
    extention : string
        Extention string to be added on the right to each filename
    Returns
    -------
    full_filenames : array of array of strings
        updated array of array with filenames
    """
    result = map(lambda x: add_full_path_and_extention_to_filenames(x, full_path, extention),
                 filenames_array_array)

    return result


def get_pascal_segmentation_image_annotation_filenames_pairs(pascal_root):
    """Return (image, annotation) filenames pairs from PASCAL VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val or trainval in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs :
        Array with filename pairs.
    """

    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(
        pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(
        pascal_root, pascal_relative_class_annotations_folder)

    pascal_images_lists_txts = get_pascal_segmentation_images_lists_txts(
        pascal_root)

    pascal_image_names = readlines_with_strip_array_version(
        pascal_images_lists_txts)

    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)

    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)

    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)

    return image_annotation_filename_pairs


def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_augmented_root):
    """ Creates a new folder in the root folder of the dataset with annotations stored in .png.
    The function accepts a full path to the root of Berkeley augmented Pascal VOC segmentation
    dataset and converts annotations that are stored in .mat files to .png files. It creates
    a new folder dataset/cls_png where all the converted files will be located. If this
    directory already exists the function does nothing. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

    Parameters
    ----------
    pascal_berkeley_augmented_root : string
        Full path to the root of augmented Berkley PASCAL VOC dataset.

    """

    import scipy.io

    def read_class_annotation_array_from_berkeley_mat(mat_filename, key='GTcls'):

        #  Mat to png conversion for http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html
        # 'GTcls' key is for class segmentation
        # 'GTinst' key is for instance segmentation
        # Credit:
        # https://github.com/martinkersner/train-DeepLab/blob/master/utils.py

        mat = scipy.io.loadmat(mat_filename, mat_dtype=True,
                               squeeze_me=True, struct_as_record=False)
        return mat[key].Segmentation

    mat_file_extension_string = '.mat'
    png_file_extension_string = '.png'
    relative_path_to_annotation_mat_files = 'dataset/cls'
    relative_path_to_annotation_png_files = 'dataset/cls_png'

    mat_file_extension_string_length = len(mat_file_extension_string)

    annotation_mat_files_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                 relative_path_to_annotation_mat_files)

    annotation_png_save_fullpath = os.path.join(pascal_berkeley_augmented_root,
                                                relative_path_to_annotation_png_files)

    # Create the folder where all the converted png files will be placed
    # If the folder already exists, do nothing
    if not os.path.exists(annotation_png_save_fullpath):

        os.makedirs(annotation_png_save_fullpath)
    else:

        return

    mat_files_names = os.listdir(annotation_mat_files_fullpath)

    for current_mat_file_name in mat_files_names:

        current_file_name_without_extention = current_mat_file_name[
            :-mat_file_extension_string_length]

        current_mat_file_full_path = os.path.join(annotation_mat_files_fullpath,
                                                  current_mat_file_name)

        current_png_file_full_path_to_be_saved = os.path.join(annotation_png_save_fullpath,
                                                              current_file_name_without_extention)

        current_png_file_full_path_to_be_saved += png_file_extension_string

        annotation_array = read_class_annotation_array_from_berkeley_mat(
            current_mat_file_full_path)

        # TODO: hide 'low-contrast' image warning during saving.
        io.imsave(current_png_file_full_path_to_be_saved, annotation_array)


def get_pascal_berkeley_augmented_segmentation_images_lists_txts(pascal_berkeley_root):
    """Return full paths to files in PASCAL Berkley augmented VOC with train and val image name lists.
    This function returns full paths to files which contain names of images
    and respective annotations for the segmentation in PASCAL VOC.

    Parameters
    ----------
    pascal_berkeley_root : string
        Full path to the root of PASCAL VOC Berkley augmented dataset.

    Returns
    -------
    full_filenames_txts : [string, string]
        Array that contains paths for train/val txts with images names.
    """

    segmentation_images_lists_relative_folder = 'dataset'

    segmentation_images_lists_folder = os.path.join(pascal_berkeley_root,
                                                    segmentation_images_lists_relative_folder)

    # TODO: add function that will joing both train.txt and val.txt into
    # trainval.txt
    pascal_train_list_filename = os.path.join(segmentation_images_lists_folder,
                                              'train.txt')

    pascal_validation_list_filename = os.path.join(segmentation_images_lists_folder,
                                                   'val.txt')

    return [
        pascal_train_list_filename,
        pascal_validation_list_filename
    ]


def get_pascal_berkeley_augmented_segmentation_image_annotation_filenames_pairs(pascal_berkeley_root):
    """Return (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset.
    Returns three dimensional array where first dimension represents the type
    of the dataset: train, val in the respective order. Second
    dimension represents the a pair of images in that belongs to a particular
    dataset. And third one is responsible for the first or second element in the
    dataset.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    Returns
    -------
    image_annotation_filename_pairs :
        Array with filename pairs.
    """

    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(
        pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(
        pascal_berkeley_root, pascal_relative_class_annotations_folder)

    pascal_images_lists_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(
        pascal_berkeley_root)

    pascal_image_names = readlines_with_strip_array_version(
        pascal_images_lists_txts)

    images_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                               pascal_images_folder,
                                                                               images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames_array_version(pascal_image_names,
                                                                                    pascal_class_annotations_folder,
                                                                                    annotations_extention)

    # Combine so that we have [(images full filenames, annotation full names), .. ]
    # where each element in the array represent train, val, trainval sets.
    # Overall, we have 3 elements in the array.
    temp = zip(images_full_names, annotations_full_names)

    # Now we should combine the elements of images full filenames annotation full names
    # so that we have pairs of respective image plus annotation
    # [[(pair_1), (pair_1), ..], [(pair_1), (pair_2), ..] ..]
    # Overall, we have 3 elements -- representing train/val/trainval datasets
    image_annotation_filename_pairs = map(lambda x: zip(*x), temp)

    return image_annotation_filename_pairs


def get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL Berkeley VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL Berkeley VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC Berkeley that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs :
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'dataset/img'
    pascal_relative_class_annotations_folder = 'dataset/cls_png'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(
        pascal_berkeley_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(
        pascal_berkeley_root, pascal_relative_class_annotations_folder)

    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)

    image_annotation_pairs = zip(images_full_names,
                                 annotations_full_names)

    return image_annotation_pairs


def get_pascal_selected_image_annotation_filenames_pairs(pascal_root, selected_names):
    """Returns (image, annotation) filenames pairs from PASCAL VOC segmentation dataset for selected names.
    The function accepts the selected file names from PASCAL VOC segmentation dataset
    and returns image, annotation pairs with fullpath and extention for those names.
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    selected_names : array of strings
        Selected filenames from PASCAL VOC that can be read from txt files that
        come with dataset.
    Returns
    -------
    image_annotation_pairs :
        Array with filename pairs with fullnames.
    """
    pascal_relative_images_folder = 'JPEGImages'
    pascal_relative_class_annotations_folder = 'SegmentationClass'

    images_extention = 'jpg'
    annotations_extention = 'png'

    pascal_images_folder = os.path.join(
        pascal_root, pascal_relative_images_folder)
    pascal_class_annotations_folder = os.path.join(
        pascal_root, pascal_relative_class_annotations_folder)

    images_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                 pascal_images_folder,
                                                                 images_extention)

    annotations_full_names = add_full_path_and_extention_to_filenames(selected_names,
                                                                      pascal_class_annotations_folder,
                                                                      annotations_extention)

    image_annotation_pairs = zip(images_full_names,
                                 annotations_full_names)

    return image_annotation_pairs


def get_augmented_pascal_image_annotation_filename_pairs(pascal_root, pascal_berkeley_root, mode=2):
    """Returns image/annotation filenames pairs train/val splits from combined Pascal VOC.
    Returns two arrays with train and validation split respectively that has
    image full filename/ annotation full filename pairs in each of the that were derived
    from PASCAL and PASCAL Berkeley Augmented dataset. The Berkley augmented dataset
    can be downloaded from here:
    http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    Consider running convert_pascal_berkeley_augmented_mat_annotations_to_png() after extraction.

    The PASCAL VOC dataset can be downloaded from here:
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    Consider specifying root full names for both of them as arguments for this function
    after extracting them.
    The function has three type of train/val splits(credit matconvnet-fcn):

        Let BT, BV, PT, PV, and PX be the Berkeley training and validation
        sets and PASCAL segmentation challenge training, validation, and
        test sets. Let T, V, X the final trainig, validation, and test
        sets.

        Mode 1::
              V = PV (same validation set as PASCAL)

        Mode 2:: (default))
              V = PV \ BT (PASCAL val set that is not a Berkeley training
              image)

        Mode 3::
              V = PV \ (BV + BT)

        In all cases:

              S = PT + PV + BT + BV
              X = PX  (the test set is uncahgend)
              T = (S \ V) \ X (the rest is training material)
    Parameters
    ----------
    pascal_root : string
        Path to the PASCAL VOC dataset root that is usually named 'VOC2012'
        after being extracted from tar file.
    pascal_berkeley_root : string
        Path to the PASCAL Berkeley VOC dataset root that is usually named 'benchmark_RELEASE'
        after being extracted from tar file.
    mode: int
        The type of train/val data split. Read the function main description for more info.
    Returns
    -------
    image_annotation_pairs : [[(string, string), .. , (string, string)][(string, string), .., (string, string)]]
        Array with filename pairs with fullnames.
    """
    pascal_txts = get_pascal_segmentation_images_lists_txts(
        pascal_root=pascal_root)
    berkeley_txts = get_pascal_berkeley_augmented_segmentation_images_lists_txts(
        pascal_berkeley_root=pascal_berkeley_root)

    pascal_name_lists = readlines_with_strip_array_version(pascal_txts)
    berkeley_name_lists = readlines_with_strip_array_version(berkeley_txts)

    pascal_train_name_set, pascal_val_name_set, _ = map(
        lambda x: set(x), pascal_name_lists)
    berkeley_train_name_set, berkeley_val_name_set = map(
        lambda x: set(x), berkeley_name_lists)

    all_berkeley = berkeley_train_name_set | berkeley_val_name_set
    all_pascal = pascal_train_name_set | pascal_val_name_set

    everything = all_berkeley | all_pascal

    # Extract the validation subset based on selected mode
    if mode == 1:

        # 1449 validation images, 10582 training images
        validation = pascal_val_name_set

    if mode == 2:

        # 904 validatioin images, 11127 training images
        validation = pascal_val_name_set - berkeley_train_name_set

    if mode == 3:

        # 346 validation images, 11685 training images
        validation = pascal_val_name_set - all_berkeley

    # The rest of the dataset is for training
    train = everything - validation

    # Get the part that can be extracted from berkeley
    train_from_berkeley = train & all_berkeley

    # The rest of the data will be loaded from pascal
    train_from_pascal = train - train_from_berkeley

    train_from_berkeley_image_annotation_pairs = \
        get_pascal_berkeley_augmented_selected_image_annotation_filenames_pairs(pascal_berkeley_root,
                                                                                list(train_from_berkeley))

    train_from_pascal_image_annotation_pairs = \
        get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                             list(train_from_pascal))

    overall_train_image_annotation_filename_pairs = \
        train_from_berkeley_image_annotation_pairs + \
        train_from_pascal_image_annotation_pairs

    overall_val_image_annotation_filename_pairs = \
        get_pascal_selected_image_annotation_filenames_pairs(pascal_root,
                                                             validation)

    return overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs


def pascal_filename_pairs_to_imageset_txt(voc_imageset_txt_path, filename_pairs, image_extension='.jpg'):
    with open(voc_imageset_txt_path, 'w') as txtfile:
        [txtfile.write(os.path.splitext(os.path.basename(file1))[0] + '\n')
         for file1, file2 in filename_pairs if file1.endswith(image_extension)]


def pascal_combine_annotation_files(filename_pairs, output_annotations_path):
    mkdir_p(output_annotations_path)
    for img_path, gt_path in filename_pairs:
        shutil.copy2(gt_path, output_annotations_path)


# ============== Ingredient 2: dataset =======================
data_pascal_voc = Experiment("dataset")


@data_pascal_voc.config
def voc_config():
    # TODO(ahundt) add md5 sums for each file
    verbose = True
    dataset_root = os.path.join(os.path.expanduser("~"), '.keras', 'datasets')
    dataset_path = dataset_root + '/VOC2012'
    # sys.path.append("tf-image-segmentation/")
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # based on https://github.com/martinkersner/train-DeepLab

    # original PASCAL VOC 2012
    # wget
    # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    # # 2 GB
    pascal_root = dataset_path + '/VOCdevkit/VOC2012'

    # berkeley augmented Pascal VOC
    # wget
    # http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
    # # 1.3 GB

    # Pascal Context
    # http://www.cs.stanford.edu/~roozbeh/pascal-context/
    # http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz
    pascal_berkeley_root = dataset_path + '/benchmark_RELEASE'
    urls = [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/33_context_labels.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/59_context_labels.tar.gz',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/33_labels.txt',
        'http://www.cs.stanford.edu/~roozbeh/pascal-context/59_labels.txt'
    ]
    filenames = ['VOCtrainval_11-May-2012.tar',
                 'benchmark.tgz',
                 'trainval.tar.gz',
                 '33_context_labels.tar.gz',
                 '59_context_labels.tar.gz',
                 '33_labels.txt',
                 '59_labels.txt'
                 ]

    md5s = ['6cd6e144f989b92b3379bac3b3de84fd',
            '82b4d87ceb2ed10f6038a1cba92111cb',
            'df034edb2c12aa7d33b42b20bb1796e3',
            '180101cfc01c71867b6686207f071eb9',
            'f85d450010762a0e1080304286ce30ed',
            '8840f5439b471aecf991ac6448b826e6',
            '993901f2d930cc038c406845f08fa082']

    combined_imageset_train_txt = dataset_path + '/combined_imageset_train.txt'
    combined_imageset_val_txt = dataset_path + '/combined_imageset_val.txt'
    combined_annotations_path = dataset_path + '/combined_annotations'

    # see get_augmented_pascal_image_annotation_filename_pairs()
    voc_data_subset_mode = 2


@data_pascal_voc.capture
def pascal_voc_files(dataset_path, filenames, dataset_root, urls, md5s):
    print(dataset_path)
    print(dataset_root)
    print(urls)
    print(filenames)
    print(md5s)
    return [dataset_path + filename for filename in filenames]


@data_pascal_voc.command
def pascal_voc_download(dataset_path, filenames, dataset_root, urls, md5s):
    zip_paths = pascal_voc_files(
        dataset_path, filenames, dataset_root, urls, md5s)
    for url, filename, md5 in zip(urls, filenames, md5s):
        path = get_file(filename, url, md5_hash=md5,
                        extract=True, cache_subdir=dataset_path)


@data_pascal_voc.command
def convert_pascal_berkeley_augmented_mat_annotations_to_png(pascal_berkeley_root):
    pascal_voc.convert_pascal_berkeley_augmented_mat_annotations_to_png(
        pascal_berkeley_root)


@data_pascal_voc.command
def pascal_voc_berkeley_combined(dataset_path,
                                 pascal_root,
                                 pascal_berkeley_root,
                                 voc_data_subset_mode,
                                 combined_imageset_train_txt,
                                 combined_imageset_val_txt,
                                 combined_annotations_path):
    # Returns a list of (image, annotation)
    # filename pairs (filename.jpg, filename.png)
    overall_train_image_annotation_filename_pairs, \
        overall_val_image_annotation_filename_pairs = \
        pascal_voc.get_augmented_pascal_image_annotation_filename_pairs(
            pascal_root=pascal_root,
            pascal_berkeley_root=pascal_berkeley_root,
            mode=voc_data_subset_mode)
    # combine the annotation files into one folder
    pascal_voc.pascal_combine_annotation_files(
        overall_train_image_annotation_filename_pairs +
        overall_val_image_annotation_filename_pairs,
        combined_annotations_path)
    # generate the train imageset txt
    pascal_voc.pascal_filename_pairs_to_imageset_txt(
        combined_imageset_train_txt,
        overall_train_image_annotation_filename_pairs
    )
    # generate the val imageset txt
    pascal_voc.pascal_filename_pairs_to_imageset_txt(
        combined_imageset_val_txt,
        overall_val_image_annotation_filename_pairs
    )


@data_pascal_voc.command
def pascal_voc_setup(filenames, dataset_path, pascal_root,
                     pascal_berkeley_root, dataset_root,
                     voc_data_subset_mode,
                     urls, md5s,
                     combined_imageset_train_txt,
                     combined_imageset_val_txt,
                     combined_annotations_path):
    # download the dataset
    pascal_voc_download(dataset_path, filenames,
                        dataset_root, urls, md5s)
    # convert the relevant files to a more useful format
    convert_pascal_berkeley_augmented_mat_annotations_to_png(
        pascal_berkeley_root)
    pascal_voc_berkeley_combined(dataset_path,
                                 pascal_root,
                                 pascal_berkeley_root,
                                 voc_data_subset_mode,
                                 combined_imageset_train_txt,
                                 combined_imageset_val_txt,
                                 combined_annotations_path)
