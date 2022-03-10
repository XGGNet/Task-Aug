import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import scipy.io as sio
import cv2

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, dataroot,mode, batch_size, num_classes, resize=256,shuffle=True, buffer_size=300):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and separated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensorFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value, different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows shuffling of the dataset.
        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes
        self.resize = resize

        # retrieve the data from the text file
        self._read_txt_file()
        # self.img_paths = self.img_paths[:-1]
        # self.data_size = self.data_size - 1

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        # self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)

        # # create dataset
        # data = tf.data.Dataset.from_tensor_slices(self.img_paths)

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.string)
        self.masks = convert_to_tensor(self.masks, dtype=dtypes.string)


        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels,self.masks))

        # patch_size = [128,128]
        # self._parse_function_train = lambda filename,label: tf.py_func(self._extract_patch, [filename,label], [tf.float32, tf.float32])
        # self._parse_function_inference = lambda filename,label: tf.py_func(self._extract_patch, [filename,label], [tf.float32, tf.float32])
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
        self.masks = []
        self.data_size = 0
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.img_paths.append(line[:-1])
                self.labels.append(line[:-1].replace('image','label'))
                self.masks.append(line[:-1].replace('image','mask'))
                self.data_size += 1


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        masks = self.masks
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        self.masks = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])
            self.masks.append(masks[i])


    def _parse_function_train(self, filename, label,mask):
        """Input parser for samples of the training set."""
        # load and pre-process the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=1)
        # img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_resized = tf.image.resize_images(img_decoded, [self.resize, self.resize])
        img_output = img_resized*1.0/255
    
        # mask = tf.where(tf.equal(img_output,tf.zeros_like(img_output)),tf.zeros_like(img_output),tf.ones_like(img_output))
        # mask = tf.where(tf.equal(img_output,tf.ones_like(img_output)),tf.zeros_like(img_output),mask)
        # mask = tf.squeeze(mask,-1)

        # condition = tf.random_uniform([], 0, 1) < 0.5
        # condition = np.random.uniform(0,1,1)<0.5
        # img_output = tf.cond(condition, lambda: self.augment(img_output),lambda: img_output)
        # if condition:
        #     img_output = self.flip(img_output)

        label_string = tf.read_file(label)
        label_decoded = tf.image.decode_png(label_string, channels=1)
        label_resized = tf.image.resize_images(label_decoded, [self.resize, self.resize],method=1)
        label_out = tf.where(label_resized>0,tf.ones_like(label_resized),tf.zeros_like(label_resized))

        mask_string = tf.read_file(mask)
        mask_decoded = tf.image.decode_png(mask_string, channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, [self.resize, self.resize],method=1)
        mask_out = tf.where(mask_resized>0,tf.ones_like(mask_resized),tf.zeros_like(mask_resized))

        # label_out = tf.cond(condition, lambda: self.augment(label_out),lambda: label_out)

        if np.random.uniform(0,1,1)<0.5:
            img_output = tf.image.flip_left_right(img_output)
            label_out = tf.image.flip_left_right(label_out)
            mask_out = tf.image.flip_left_right(mask_out)
        
        if np.random.uniform(0,1,1)<0.5:
            img_output = tf.image.flip_up_down(img_output)
            label_out = tf.image.flip_up_down(label_out)
            mask_out = tf.image.flip_up_down(mask_out)
        
        label_out = tf.squeeze(label_out,-1)
        label_out = tf.one_hot(label_out, self.num_classes)

        mask_out = tf.squeeze(mask_out,-1)

        return img_output, label_out, mask_out

    def _parse_function_inference(self, filename, label,mask):
        """Input parser for samples of the validation/test set."""
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=1)
        # img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_resized = tf.image.resize_images(img_decoded, [self.resize, self.resize])
        img_output = img_resized*1.0/255

        # mask = tf.where(tf.equal(img_output,tf.zeros_like(img_output)),tf.zeros_like(img_output),tf.ones_like(img_output))
        # mask = tf.where(tf.equal(img_output,tf.ones_like(img_output)),tf.zeros_like(img_output),mask)
        # mask = tf.squeeze(mask,-1)

        # condition = tf.random_uniform([], 0, 1) < 0.5
        # img_output = tf.cond(condition, lambda: self.augment(img_output),lambda: img_output)

        label_string = tf.read_file(label)
        label_decoded = tf.image.decode_png(label_string, channels=1)

        label_resized = tf.image.resize_images(label_decoded, [self.resize, self.resize],method=1)
        # label_resized = label_decoded

        label_out = tf.squeeze(tf.where(label_resized>0,tf.ones_like(label_resized),tf.zeros_like(label_resized)),-1)
        # label_out = tf.cond(condition, lambda: self.augment(label_out),lambda: label_out)

        label_out = tf.one_hot(label_out, self.num_classes)

        mask_string = tf.read_file(mask)
        mask_decoded = tf.image.decode_png(mask_string, channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, [self.resize, self.resize],method=1)
        mask_resized = tf.where(mask_resized>0,tf.ones_like(mask_resized),tf.zeros_like(mask_resized))

        mask_output = tf.squeeze(mask_resized,-1)

        return img_output, label_out, mask_output,filename

    def augment(self, x):
        # add more types of augmentations here
        augmentations = [self.flip]
        for f in augmentations:
            x = tf.cond(tf.random_uniform([], 0, 1) < 0.25, lambda: f(x), lambda: x)
            # x = f(x)
            
        return x

    def flip(self, x):
        """Flip augmentation
        Args:
            x: Image to flip
        Returns:
            Augmented image
        """
        x = tf.image.random_flip_left_right(x)

    # def _extract_patch(self, filename,label):

    #     # mat_content = sio.loadmat(filename)
    #     # image = mat_content['image']
    #     # image = image[34:-35,16:-17,:] # [18:-19,1:,:] is 192*192
    #     # label = mat_content['label']
    #     # label = label[34:-35,16:-17,0]
    #     # filename=str(filename)
    #     # print(filename)
    #     print(filename)
    #     print(label)
    #     img_string = tf.read_file(filename)
    #     img_decoded = tf.image.decode_png(img_string, channels=1)
    #     label_string = tf.read_file(label)
    #     label_decoded = tf.image.decode_png(label_string, channels=1)

    #     # print(filename)
    #     # image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)*1.0/255
    #     # print(image.shape)
    #     # label = cv2.imread(filename.replace('image','label'),cv2.IMREAD_GRAYSCALE)
    #     # label = np.where(label>0,1,0).astype(np.float32)
    #     # label = self._label_decomp(label)
    #     # mask = np.ones_like(image)
    #     # mask = np.where(image==0,0,mask)
    #     # mask = np.where(image==1,0,mask) 
    #     return img_decoded, label_decoded

    # def _label_decomp(self, label_vol):
    #     """
    #     decompose label for softmax classifier
    #     original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    #     this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    #     numpy version of tf.one_hot
    #     """
    #     one_hot = []
    #     for i in range(self.num_classes):
    #         _vol = np.zeros(label_vol.shape)
    #         _vol[label_vol == i] = 1
    #         one_hot.append(_vol)

    #     return np.stack(one_hot, axis=-1)
