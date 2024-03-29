import tensorflow as tf
import numpy as np
import os
# from matplotlib import pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import skimage as sk
from skimage import transform


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, dataroot, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):

        """Create a new ImageDataGenerator.
        Receives a path string to a text file, where each line has a path string to an image and
        separated by a space, then with an integer referring to the class number.

        Args:
            txt_file: path to the text file.
            mode: either 'training' or 'validation'. Depending on this value, different parsing functions will be used.
            batch_size: number of images per batch.
            num_classes: number of classes in the dataset.
            shuffle: wether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: number of images used as buffer for TensorFlows shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.dataroot = dataroot
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        if mode == 'training':
            self._read_txt_file()

        elif mode == 'inference':
            self._read_txts_file()


        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists together
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(os.path.join(self.dataroot, items[0]))
                # print (os.path.join(self.dataroot, items[0]))
                self.labels.append(int(int(items[1]))) # 医学图像
                # self.labels.append(int(int(items[1]-1))) #自然图像

    def _read_txts_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        for i in range(len(self.txt_file)):
            print (self.txt_file)
            with open(self.txt_file[i], 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(' ')
                    self.img_paths.append(os.path.join(self.dataroot, items[0]))
                    # print (os.path.join(self.dataroot, items[0]))
                    self.labels.append(int(int(items[1])))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        
        one_hot = tf.one_hot(label, self.num_classes)

        # load and pre-process the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])

        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        # Data augmentation comes here, with a chance of 50%
        img_bgr = tf.cond(tf.random_uniform([], 0, 1) < 0.5, lambda: self.augment(img_bgr),lambda: img_bgr)

        return img_bgr, one_hot

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])

        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def augment(self, x):
        # add more types of augmentations here
        augmentations = [self.flip]
        for f in augmentations:
            x = tf.cond(tf.random_uniform([], 0, 1) < 0.25, lambda: f(x), lambda: x)
            
        return x

    def flip(self, x):
        """Flip augmentation
        Args:
            x: Image to flip
        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x)
        # x = tf.image.random_flip_up_down(x)

        return x
