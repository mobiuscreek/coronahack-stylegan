"""Image processing tools adapted from 
https://github.com/NVlabs/stylegan2/blob/master/dataset_tool.py
"""
import os
import logging
import numpy as np
import tensorflow as tf
import glob

from PIL import Image



class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images,
                 print_progress=True, progress_interval=10):
        self.tfrecord_dir = tfrecord_dir
        self.tfr_prefix = os.path.join(
            self.tfrecord_dir, os.path.basename(
                self.tfrecord_dir))
        self.expected_images = expected_images
        self.cur_images = 0
        self.shape = None
        self.resolution_log2 = None
        self.tfr_writers = []
        self.print_progress = print_progress
        self.progress_interval = progress_interval

        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    # Note: Images and labels must be added in shuffled order.
    def choose_shuffled_order(self):
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print(
                '%d / %d\r' %
                (self.cur_images,
                 self.expected_images),
                end='',
                flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + \
                    '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(
                    tf.python_io.TFRecordWriter(
                        tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] +
                       img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def list_image_filenames(image_dir):
    """Recurse through image_dir, return paths to image files"""
    matches = []
    for root, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith('jpg') or filename.lower().endswith('png') or filename.lower().endswith('jpeg'):
                matches.append(os.path.join(root, filename))
    return matches

def tfrecords_from_images(tfrecord_dir, image_dir, shuffle):
    print('Loading images from "%s"' % image_dir)
    image_filenames = list_image_filenames(image_dir)
    if len(image_filenames) == 0:
        logging.error('No input images found')

    img = np.asarray(Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        logging.error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        logging.error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        logging.error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :]  # HW => CHW
            else:
                img = img.transpose([2, 0, 1])  # HWC => CHW
            tfr.add_image(img)

def image_resize(filename, directory=None, size=(256,256)):
    """Accepts a filename of an image.
    Converts to grayscale and resizes images.
    Stores results in a directory if specified.
    """

    image = Image.open(filename).convert('L')
    im_res = image.resize(size)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = filename.split('/')[-1]
        im_res.save(f'{directory}/{filename}')
    return im_res


