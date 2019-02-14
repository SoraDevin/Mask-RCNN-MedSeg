"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib.image
import glob
import scipy.misc
from PIL import Image
import imgaug
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.getcwd()
# Training data location
MAMOGRAM_IMAGE_DIR = "/scans/image/"
MAMOGRAM_MASK_DIR = "/scans/mask/"

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEMO_SAVE_DIR = "demos/"

############################################################
#  Configurations
############################################################


class MamogramConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mamogram"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + lesion
    
    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 190

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 30

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9


############################################################
#  Dataset
############################################################

class MamogramDataset(utils.Dataset):

    def load_mamogram(self, subset):
        """This method loads the actual image
        subset is either "train" or "val" depending on whether the image is part of the training or validation datasets 
        """
        # Add classes. We have only one class to add.
        # These are the things that will be segmented
        self.add_class("mamogram", 1, "lesion")

        #list all the files in the directory with the mamogram images
        files = os.listdir(ROOT_DIR + MAMOGRAM_IMAGE_DIR + subset + "/")

        for fname in files:
            self.add_image("mamogram", image_id=fname, path=ROOT_DIR + MAMOGRAM_IMAGE_DIR + subset +"/"+ fname, subset=subset, fname=fname)


    def load_mask(self, image_id):
        """load the instance masks for an image.
        Returns:
        a tuple containing:
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.

        use dtype=np.int32
        """
        info = self.image_info[image_id]
        fname = info['fname']

        files = glob.glob(ROOT_DIR + MAMOGRAM_MASK_DIR + info['subset'] + fname[0:-4] + "*")

        masks = []
        for i in range(0, len(files)):
            data = skimage.io.imread(files[i])
            #if images are single dimension, expand to fill 3 dimensions
            #singleMask = np.expand_dims(data, axis=2)
            #singleMask = np.stack((data,)*3, axis=-1)
            if data.ndim != 1:
                data = skimage.color.rgb2gray(data)
            # If has an alpha channel, remove it for consistency
            if data.shape[-1] == 4:
                data = data[..., :3]
            singleMask = data
            if i == 0:
                masks = np.zeros((singleMask.shape[0], singleMask.shape[1], len(files)))
            masks[:,:,i] = singleMask

        instanceMaskMap = np.array(np.ones([masks.shape[-1]], dtype=np.int32)) #this is VERY important: array of class ids in the order that they appear in bigdata
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return (masks.astype(np.bool), instanceMaskMap)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
		Taken from utils.py, any refinements we need can be done here
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = MamogramDataset()
    dataset_train.load_mamogram("/train/")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = MamogramDataset()
    dataset_val.load_mamogram("/val/")
    dataset_val.prepare()
    
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    aug = iaa.Sequential([
        iaa.OneOf([iaa.Fliplr(0.5),
                   iaa.Flipud(0.5),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)])
        # iaa.Multiply((0.8, 1.2)),
        # iaa.GaussianBlur(sigma=(0.0, 2.0))
    ])

    print("Training network layers")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                augmentation=aug,
                layers='all')

def segment(model, imPath):
    
    image = skimage.io.imread(imPath)
    #image = np.expand_dims(image, axis=2)
    fname = imPath.split('/')[-1]
    mrcnnData = model.detect([image], verbose=1)[0]
    # documentation for model.detect:
    """Runs the detection pipeline.

    images: List of images, potentially of different sizes.

    Returns a list of dicts, one dict per image. The dict contains:
    rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    class_ids: [N] int class IDs
    scores: [N] float probability scores for the class IDs
    masks: [H, W, N] instance binary masks
    """
    
    #Merge masks and save as single file
    masks = mrcnnData['masks']
    for i in range(0, masks.shape[2]):
        #iterate through the masks
        maskSingle = np.squeeze(masks[:, :, i])
        file_name = DEMO_SAVE_DIR + "demo_mask_" + str(i) + "_" + fname + "_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        

        scipy.misc.imsave(file_name, maskSingle.astype(np.int64)) 


    print(mrcnnData)
    print("&&&&&&&&&&&: "+str(mrcnnData['rois']))
    print("&&&&&&&&&&&: "+str(mrcnnData['class_ids']))
    print("&&&&&&&&&&&: "+str(mrcnnData['scores']))

    return

def segmentWrapper(model, directory):
    """wrapper function for segment to take many images as an input, calls segment() on everything in the directory"""
    files = os.listdir(directory)
    for f in files:
        segment(model, directory + '/' + f)

def overlayResult(image, mask):
	"""Function to overlay segmentation mask on an image.
	usage: image_var = overlayResult(image, dict['masks'] || masks_var)
	
	image: RGB or grayscale image [height, width, 1 || 3].
	mask: segmentation mask [height, width, instance_count]
	
	returns resulting image.
	"""
	# Image is already in grayscale so we skip converting it
	# May need to create 3 dimensions if single dimension image though so
	# will add this as a placeholder
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		#collapse masks into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		overlay = np.where(mask, image, gray).astype(np.uint8)
	else:
		overlay = gray.astype(np.uint8)
		
	return overlay

# def compareResultWithTruth(model, image_path):
    # image = skimage.io.imread(image_path)
    # r = model.detect([image], verbose=1)[0]
    # overlay = overlayResult(image, r['masks'])
    # file_name = "overlay_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    # skimage.io.imsave(file_name, overlay)
	
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect breast lesions.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'segment'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar='/path/to/image',
                        help="Path to image file for segmentation")
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = MamogramConfig()
    else:
        class InferenceConfig(MamogramConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "segment":
        if os.path.isdir(args.image):
            segmentWrapper(model, args.image)
        else:
            segment(model, args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'segment'".format(args.command))
