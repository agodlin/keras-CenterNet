import numpy as np
import tensorflow as tf
import random
from yolo.utils import get_random_data_magic, read_data_file, get_data_magic
from yolo.yolo import preprocess_true_boxes, yolo_input_shape, yolo_features_output
from yolo.preprocessing import yolo_image_normalization_int, yolo_image_normalization_flaot
import keras
def create_data_generator(args, list_path, val=False) -> tf.keras.utils.Sequence:
    data_provider = FDDataGenerator(list_path=list_path,
                                 images_relative_path=args.images_root_path,
                                 batch_size=args.batch_size,
                                 image_size=args.image_size,
                                 input_channels=args.input_channels,
                                 args=args, val=val)
    return data_provider

class FDDataGenerator(keras.utils.Sequence):

    def __init__(self,list_path,
                 images_relative_path,
                 batch_size,
                 image_size,
                 input_channels,
                 args, val=False):
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_channels = input_channels
        self.val = val
        self.args = args

        images_list, labels_list, images_orientation, self._subjects = read_data_file(list_path, images_relative_path)
        data_list = list(zip(images_list, labels_list, images_orientation))
        if val:
            random.seed(17)
            #random.shuffle(data_list)
            data_list = data_list
        else:
            if args.random_seed > 0:
                random.seed(args.random_seed)
            random.shuffle(data_list)
            if args.train_list_clip > 0:
                data_list = data_list[:args.train_list_clip]
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list) // self.batch_size

    def __getitem__(self,index):
        batch = []
        labels = []
        for idx in range(index*self.batch_size,((index+1)*self.batch_size)):

            img_r, label = self.get_image(idx)
            #noised_image = img_r + np.random.normal(0,0.02,img_r.shape)
            batch.append(img_r)
            #batch.append(noised_image)
            #labels.append(label)
            labels.append(label)

        return np.array(batch), np.array(labels)

    def on_epoch_end(self):
        random.shuffle(self.data_list)

    def get_image(self, image_index):
        filename = self.data_list[image_index][0]
        label = self.data_list[image_index][1]
        image_orientation = self.data_list[image_index][2]
        image_normalize = yolo_image_normalization_flaot
        if self.args.train_image_normalization == 'int':
            image_normalize = yolo_image_normalization_int
        if self.val:
            image, label = get_data_magic(filename, label, image_orientation, yolo_input_shape, image_normalize=image_normalize)
        else:
            image, label = get_random_data_magic(filename, label, image_orientation, yolo_input_shape, image_normalize=image_normalize)

        return image, label

    def num_classes(self):
        return 1