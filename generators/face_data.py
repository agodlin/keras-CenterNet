"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
from generators.common import Generator
import numpy as np
import random
import os
from six import raise_from
import xml.etree.ElementTree as ET
from yolo.yolo_utils import get_random_data_magic, read_data_file, get_data_magic, load_image
face_classes = {
    'face': 0,
}

class FaceGenerator(Generator):
    """
    Generate data for a Pascal VOC dataset.

    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
            self,
            file_path,
            images_relative_path,
            classes=face_classes,
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        """
        Initialize a Pascal VOC data generator.

        Args:
            data_dir: the path of directory which contains ImageSets directory
            set_name: test|trainval|train|val
            classes: class names tos id mapping
            image_extension: image filename ext
            skip_truncated:
            skip_difficult:
            **kwargs:
        """
        self.classes = classes
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        # class ids to names mapping
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        images_list, labels_list, images_orientation, self._subjects = read_data_file(file_path, images_relative_path)
        data_list = list(zip(images_list, labels_list, images_orientation))
        random.shuffle(data_list)
        self.data_list = data_list[:10]
        super(FaceGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.data_list)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        image = load_image(self.data_list[image_index][0],self.data_list[image_index][2])[0]
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        image = load_image(self.data_list[image_index][0],self.data_list[image_index][2])[0]
        return np.stack([image,image,image], axis=2)

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        label = self.data_list[image_index][1]
        if len(label) > 0:
            class_labels = label[:,-1].flatten()
            bboxes = label[:,:-1]
        else:
            class_labels = np.empty((0,), dtype=np.int32)
            bboxes = np.empty((0, 4))
        annotations = {'labels': class_labels, 'bboxes':bboxes}
        return annotations


if __name__ == '__main__':
    from augmentor.misc import MiscEffect
    from augmentor.color import VisualEffect

    misc_effect = MiscEffect(border_value=0)
    visual_effect = VisualEffect()

    generator = FaceGenerator(
        r'D:\datasets\test6_f450.txt',
        r'\\ger\ec\proj\ha\RSG',
        skip_difficult=True,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        batch_size=1
    )
    for inputs, targets in generator:
        print('hi')