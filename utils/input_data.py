"""Functions for reading input data (image (dicom) and label (txt))."""

import os
import numpy as np
#from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import shape_model_func
import pydicom
from operator import itemgetter


class DataSet(object):
  def __init__(self,
               names,
               images,
               labels,
               shape_params,
               pix_dim):
    assert len(images) == labels.shape[0], ('len(images): %s labels.shape: %s' % (len(images), labels.shape))
    self.num_examples = len(images)
    self.names = names
    self.images = images
    self.labels = labels
    self.shape_params = shape_params
    self.pix_dim = pix_dim


def get_file_list(txt_file):
    """
    Get a list of filenames.

    Args:
        txt_file: Name of a txt file containing a list of filenames for the images.

    Returns:
        filenames: A list of filenames for the images.

    """
    with open(txt_file) as f:
        filenames = f.read().splitlines()
    return filenames



def extract_image(filename):
    """ Read in the directory of a single subject and return a numpy array
    Extract the image into a 3D numpy array [x, y, z].

      Args:
        filename: Path and name of dicom file.

      Returns:
        data: A 3D numpy array [x, y, z]
        pix_dim: voxel spacings

     """
    patient_path = os.path.join(filename)
    patient_image_paths = [os.path.join(patient_path, slice_name) for slice_name in os.listdir(patient_path)]
    patient_images = [pydicom.read_file(patient_slice_path) for patient_slice_path in patient_image_paths]
    # some of the slices are not valid and must be excluded
    patient_slices = [patient_slice for patient_slice in patient_images if 0 <= int(patient_slice.InstanceNumber) < len(patient_images)]
    dicom_image = sorted(patient_slices, key=lambda x: int(x.InstanceNumber))
    volume = [dicom_to_hounsfield_units(dvt_slice) for dvt_slice in dicom_image]
    data = np.stack( volume, axis=0 )
    pix_dim = np.float(patient_slices[0].PixelSpacing[0]), np.float(patient_slices[0].PixelSpacing[1]), np.float(patient_slices[0].SliceThickness)
    return data, pix_dim


def dicom_to_hounsfield_units(dicom_image):
    """ Transforms the pixel values of a DICOM file into their value in the Hounsfield scale (quantitative scale for
    describing radiodensity).

    :param dicom_image: DICOM image
    :type dicom_image: `pydicom.dataset.FileDataset`
    :return: the DICOM image in Hounsfield Units
    :rtype: `numpy.ndarray`
    """
    intercept = np.float(dicom_image.RescaleIntercept)
    slope = np.float(dicom_image.RescaleSlope)

    dicom_image = dicom_image.pixel_array.astype(np.float32) #float64
    return (dicom_image * slope + intercept).astype(np.int16)



def extract_label(filename):
  """Extract the labels (landmark coordinates) into a 2D float64 numpy array.

  Args:
    filename: Path and name of txt file containing the landmarks. One row per landmark.

  Returns:
    labels: a 2D float64 numpy array. [landmark_count, 3]
  """
  with open(filename) as f:
    labels = np.empty([0, 3], dtype=np.float64)
    for line in f:
        #labels = np.vstack((labels, np.asarray(map(float, line.split()))))
        labels2 = np.fromiter(map(float,line.split()), dtype=np.float64)
        labels = np.vstack((labels, labels2))

  return labels




def select_label(labels, landmark_unwant):
  """Unwanted landmarks are removed.
     Remove topHead (landmark index 0).
     Remove left or right ventricle (landmark index (6,7) or (8,9)).
     Remove mid CSP (landmark index 13).
     Remove left and right eyes (landmark index 14 and 15).
  Args:
    labels: a 2D float64 numpy array.
    landmark_unwant: indices of the unwanted landmarks
  Returns:
    labels: a 2D float64 numpy array.
  """
  removed_label_ind = list(landmark_unwant)
  labels = np.delete(labels, removed_label_ind, 0)
  return labels



#  extract_all_image_and_label
def generate_batch_image_and_label(data_dir,
                                   label_dir,
                                   file_list,
                                   landmark_count,
                                   landmark_unwant,
                                   shape_model,
                                   batch_size):
  """Load the input images and landmarks and rescale to fixed size.

  Args:
    file_list: txt file containing list of filenames of images
    data_dir: Directory storing images.
    label_dir: Directory storing labels.
    landmark_count: Number of landmarks used (unwanted landmarks removed)
    landmark_unwant: discard these landmarks
    shape_model: structure containing the shape model

  Returns:
    filenames: list of patient id names
    images: list of img_count 4D numpy arrays with dimensions=[width, height, depth, 1]. Eg. [324, 207, 279, 1]
    labels: landmarks coordinates [img_count, landmark_count, 3]
    shape_params: PCA shape parameters [img_count, shape_param_count]
    pix_dim: mm of each voxel. [img_count, 3]

  """
  #while True:
  filenames = get_file_list(file_list)
  file_count = len(filenames)
  batchcount = 0
  images = []
  #labels = []
  #pix_dims = []

  labels = np.zeros((file_count, landmark_count, 3), dtype=np.float64)
  pix_dim = np.zeros((file_count, 3))


  for i in range(len(filenames)):
      filename = filenames[i]
      print("Loading image {}/{}: {}".format(i+1, len(filenames), filename))
      # load image
      img, pix_dim[i] = extract_image(os.path.join(data_dir, filename))
      # load landmarks and remove unwanted ones. Labels already in voxel coordinate
      label = extract_label(os.path.join(label_dir, filename+'_ps.txt'))
      label = select_label(label, landmark_unwant)
      # Store extracted data
      images.append(np.expand_dims(img, axis=3))
      #labels.append(label)
      #pix_dims.append(pix_dim)
      batchcount += 1
      labels[i, :, :] = label

      if batchcount == batch_size:
          shape_params = shape_model_func.landmarks2b(labels, shape_model)
          return (filename, images, labels, shape_params, pix_dim)
          
