import numpy as np
import cv2
from PIL import Image
import random
from yolo.preprocessing import lut_int8
import os
from functools import partial

def autodetect_size(num_pixels):

    supported_resolutions = [(1920,1080), (1280,720), (1024,768), (800,600), (640,480), (320,240),
                             (1152,1152), (720,720), (1024, 1024)]

    for res in supported_resolutions:
        if num_pixels == res[0] * res[1]:
            return res

    return None, None


def red_w10_buffer(filename, width=None, height=None):
    with open(filename, 'rb') as infile:
        data = infile.read()

    if width is None or height is None:
        width, height = autodetect_size(len(data) * 8 / 10)

    if width is None or height is None:
        width, height = autodetect_size(len(data) / 2)

    if width is None or height is None:
        width, height = autodetect_size(len(data) / (83 / 54))

    if width is None or height is None:
        raise RuntimeError('Could not autodetect image size')

    bytes_to_read = int(width*height*(10/8))
    data = data[0:bytes_to_read]
    data = np.frombuffer(data, np.uint8)
    data = np.reshape(data,(-1,5)).astype(np.uint16)
    return data, width, height


def read_w10(filename, width=None, height=None):
    """
    Read a w10 IR file.

    :param filename: Path to the w10 file.
    :param width, height: Optional, specify the resolution of the image. If not specified, the image size
      is auto-detected from the file size using autodetect_size() (if the resolution cannot be determined,
      a RuntimeError is raised).
    :return: The w10 image, as a uint16 NumPy array.
    """
    data, width, height = red_w10_buffer(filename, width, height)

    buff = data[:,0:4]
    extra = data[:,4]

    out_image = np.zeros((data.shape[0],4), dtype='uint16')

    for j in range(4):
        first_8_bits = buff[:,j] << 2
        last_2_bits = (extra >> (j*2) & 3)
        out_image[:,j] = first_8_bits + last_2_bits

    image = np.reshape(out_image, (height, width))
    return image

def extract_red_channel(input_im, bayer_pattern='grbg'):
    """
    Extract and return the red channel from a Bayer image.

    :param input_im: The input Bayer image.
    :param bayer_pattern: The Bayer pattern of the image, either 'rggb' or 'bggr'.
    :return: The extracted channel, of the same type as the image.
    """
    d = {'rggb':(0,0), 'bggr':(1,1), 'grbg': (0,1), 'girg':(1,0)}
    assert bayer_pattern in d, 'Invalid Bayer pattern \'{}\''.format(bayer_pattern)

    red_idx = d[bayer_pattern][0]
    red_idy = d[bayer_pattern][1]
    im = input_im[red_idx::2, red_idy::2, ...]

    return im

def extract_channel(input_im, x_channel=0, y_channel=0):
    """
    Extract and return the red channel from a Bayer image.

    :param input_im: The input Bayer image.
    :param bayer_pattern: The Bayer pattern of the image, either 'rggb' or 'bggr'.
    :return: The extracted channel, of the same type as the image.
    """
    assert x_channel in (0, 1), 'Invalid Bayer X channel'
    assert y_channel in (0, 1), 'Invalid Bayer X channel'

    red_idx = x_channel
    red_idy = y_channel
    im = input_im[red_idx::2, red_idy::2, ...]

    return im

extract_f450_w10_red_channel = partial(extract_channel, x_channel=0, y_channel=1)
extract_f450_w10_green_channel = partial(extract_channel, x_channel=0, y_channel=0)

def read_w10_red(filename, width=None, height=None, upsample=1):
    data, width, height = red_w10_buffer(filename, width, height)
    buff = data[:,0:4]
    extra = data[:,4]
    out_image = np.zeros((data.shape[0]//2//upsample, 2//upsample), dtype='uint16')
    out_image_width = width//2//upsample
    for r in range(0,height,2*upsample):
        for j in range(2//upsample):
            first_8_bits = buff[r*out_image_width:(r+1)*out_image_width, j*2] << 2
            last_2_bits = (extra[r*out_image_width:(r+1)*out_image_width] >> (j*2*2) & 3)
            i = r // 2 // upsample
            out_image[(i*out_image_width):(i+1)*out_image_width, j] = first_8_bits + last_2_bits

    image = np.reshape(out_image, (height//2//upsample, width//2//upsample))
    return image

def read_w10_red_f450(filename, width=None, height=None):
    img_w10 = read_w10(filename, width=width, height=height)
    img_w10_red_channel = extract_f450_w10_red_channel(img_w10)
    return img_w10_red_channel

def read_w10_green_f450(filename, width=None, height=None):
    img_w10 = read_w10(filename, width=width, height=height)
    img_w10_red_channel = extract_f450_w10_green_channel(img_w10)
    return img_w10_red_channel


def load_image(path, image_orientation):
    d = {'landscape': (None, None), 'portrait': (1080, 1920)}
    if path.endswith('.w10'):
        width, height = d[image_orientation]  if image_orientation in d.keys() else (1080, 1920)
        image = read_w10_red_f450(path, width, height)
        image = lut_int8(image)
        return image, 2, 255
    elif path.endswith('npy'):
        image = np.load(path)
        return image, 1, 255
    else:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image, 1, 255

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def rand2(a, b):
    random.uniform(a,b)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def bb_inter_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return interArea

def gamma_correction(image, correction=1.0):
    image = np.sign(image) * np.abs(image) ** correction
    return image

def get_data_magic(image_path, box, image_orientation, input_shape, image_normalize, max_boxes=20):
    debug = False
    image, scale, normalized_factor = load_image(image_path, image_orientation)
    ih, iw = image.shape[:2]
    h, w = input_shape
    box = box / scale
    show_image(debug, image, 0, 1, 'orig', image_path, 0, 0, box)

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx, dy = ((w - nw) // 2, (h - nh) // 2)

    if nw > iw or nh > ih:
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    else:
        resize_algo_nearest = rand() < 0.8
        interpolation = cv2.INTER_AREA
        if resize_algo_nearest:
            interpolation = cv2.INTER_NEAREST
        image = cv2.resize(image, (nw, nh), interpolation=interpolation)

    #show_image(debug, image, 0, 1, 'resize', image_path, iw, 0)
    image = image.astype(np.float32)
    # place image
    new_image = Image.fromarray(np.zeros((w, h), dtype=np.float32))
    new_image.paste(Image.fromarray(image), (dx, dy))
    image = np.asarray(new_image)
    show_image(debug, image, 0, 1, 'crop',image_path, iw, 0)
    #show_image(debug, image, 0, 1, 'noise',image_path, iw+nw*2, 0)
    image_data = image

    image_data = image_normalize(image_data)

    #show_image(debug, image_data, 128, 1, 'train',image_path, iw+nw*2, nh)
    image_data = np.expand_dims(image_data, axis=2)
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    j = 0

    min_face_size = 14
    max_face_size = 100

    if len(box)>0:
        # np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy

        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]

        box = box[np.logical_and(box_w > min_face_size//2, box_h > min_face_size//2)] # discard invalid box
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w < max_face_size, box_h < max_face_size)]  # discard invalid box
        if len(box)>max_boxes:
            box = box[:max_boxes]

        for i in range(len(box)):
            box_w = box[i, 2] - box[i, 0]
            box_h = box[i, 3] - box[i, 1]
            inter_area = bb_inter_area(box[i, 0:4], [max(dx,0),max(dy,0),min(max(dx,0)+nw, w),min(max(dy,0)+nh,h)])
            inter_area_part = inter_area/(box_h*box_w)
            if inter_area_part > 0.5:
                box[i, 0] = max(0,box[i, 0])
                box[i, 1] = max(0, box[i, 1])
                box[i, 2] = min(w,box[i, 2])
                box[i, 3] = min(h,box[i, 3])
                box_data[j] = box[i]
                j+=1
    show_image(debug, image_data, 128, 1, 'train_box', image_path, iw, h, box_data[:3,:], True)
    return image_data, box_data

def get_random_data_magic(image_path, box, image_orientation, input_shape, max_boxes=20, jitter=.1, distor_image=True, image_normalize=None):
    '''randomize preprocessing for real-time data augmentation'''
    debug = False
    image, scale, normalized_factor = load_image(image_path, image_orientation)
    # print(image_path, box.shape)
    ih, iw = image.shape[:2]
    h, w = input_shape
    box = box / scale
    # print(box)
    # resize image
    new_ar = iw/ih * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)

    min_scale = 0.25
    max_scale = 2

    min_face_size = 14
    max_face_size = 70

    if len(box) > 0:
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        max_value = max(np.max(box_w), np.max(box_h))
        min_value = min(np.min(box_w), np.min(box_h))
        max_side = max(iw,ih)
        min_side = min(iw,ih)
        max_scale = max_face_size * max_side / (w * max_value+1)
        min_scale = min_face_size * min_side / (w * min_value+1)
        max_scale = min(max_scale, 4)
        min_scale = max(min_scale, 0.25)
    min_face_size = 7
    max_face_size = 100
    show_image(debug, image, 0, 1, 'orig', image_path, 0, 0, box)
    scale = rand(min_scale, max_scale)
    # print(scale,min_scale,max_scale, iw,ih)
    scale = min(scale, 4)
    scale = max(scale, 0.25)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)

    if nw > iw or nh > ih:
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    else:
        resize_algo_nearest = rand() < 0.8
        interpolation = cv2.INTER_AREA
        if resize_algo_nearest:
            interpolation = cv2.INTER_NEAREST
        image = cv2.resize(image, (nw, nh), interpolation=interpolation)

    #show_image(debug, image, 0, 1, 'resize', image_path, iw, 0)
    image = image.astype(np.float32)
    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))

    new_image = Image.fromarray(np.zeros((w, h), dtype=np.float32))
    new_image.paste(Image.fromarray(image), (dx, dy))
    image = np.asarray(new_image)
    show_image(debug, image, 0, 1, 'crop',image_path, iw, 0)
    # flip image or not
    flip = rand()<.5
    if flip:
        image = cv2.flip(image, flipCode=1)
    # add gaussian noise
    noise_coeff = rand(0, 0.3)
    image_noise = noise_coeff * np.random.randn(h, w).astype(image.dtype) * np.sqrt(image)
    image = np.clip(image + image_noise, 0.0, normalized_factor)
    #show_image(debug, image, 0, 1, 'noise',image_path, iw+nw*2, 0)
    image_data = (image / normalized_factor)

    if distor_image:
        image_data = gamma_correction(image_data, random.uniform(0.75, 1.25))
    else:
        image_data= image
    image_data = image_data * normalized_factor
    image_data = image_normalize(image_data)

    #show_image(debug, image_data, 128, 1, 'train',image_path, iw+nw*2, nh)
    image_data = np.expand_dims(image_data, axis=2)
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    j = 0
    if len(box)>0:
        # np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip:
            box[:, [0,2]] = w - box[:, [2,0]]
            dx = w - (dx + nw)
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]

        box = box[np.logical_and(box_w > min_face_size//2, box_h > min_face_size//2)] # discard invalid box
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w < max_face_size, box_h < max_face_size)]  # discard invalid box
        if len(box)>max_boxes:
            box = box[:max_boxes]

        for i in range(len(box)):
            box_w = box[i, 2] - box[i, 0]
            box_h = box[i, 3] - box[i, 1]
            inter_area = bb_inter_area(box[i, 0:4], [max(dx,0),max(dy,0),min(max(dx,0)+nw, w),min(max(dy,0)+nh,h)])
            inter_area_part = inter_area/(box_h*box_w)
            if inter_area_part > 0.5:
                box[i, 0] = max(0,box[i, 0])
                box[i, 1] = max(0, box[i, 1])
                box[i, 2] = min(w,box[i, 2])
                box[i, 3] = min(h,box[i, 3])
                box_data[j] = box[i]
                j+=1
    show_image(debug, image_data, 0.5, 255, 'train_box', image_path, iw, h, box_data[:3,:], True)
    return image_data, box_data


def show_image(debug, image_data, b, a, name, image_path, x=100, y=100, box_data=[], wait=False):
    if not debug:
        return
    color = ((image_data + b)*a).astype(np.uint8)
    color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)

    for i in range(len(box_data)):
        box = np.array(box_data[i])
        cv2.rectangle(color, tuple(box[:2].astype(np.int)), tuple(box[2:4].astype(np.int)),
                      (0, 255, 0), 3)
    winname = name
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, color)
    if name == 'train_box':
        pass
        #write_image_and_json(color, image_path)
    if wait:
        cv2.waitKey()

def write_image_and_json(color, image_path):
    import hashlib
    uname = hashlib.md5(color).hexdigest() + '.jpg'
    d = r'D:\datasets\rect_test'
    cv2.imwrite(os.path.join(d, uname), color)
    with open(os.path.join(d, uname) + '.txt', 'w') as f_w:
        f_w.writelines(image_path)
        print(image_path)

def read_data_file(list_path, images_relative_path=None):
    images_list = []
    labels_list = []
    images_orientation = []
    lines = open(list_path).readlines()
    for line in lines:
        image_path, label, image_orientation = decode_data(line.strip(), images_relative_path)
        if image_path is None or label is None:
            continue
        images_list += [image_path]
        labels_list += [label]
        images_orientation += [image_orientation]
    return images_list, labels_list, images_orientation, len(images_list)


def decode_data(data, images_relative_path):
    image_path, label, image_orientation = parse_image_label(data)
    if images_relative_path is not None:
        image_path = os.path.join(images_relative_path, image_path)
    return image_path, label, image_orientation


def parse_image_label(image_line):
    data = image_line.strip().split()
    file_path = data[0]
    image_orientation = data[-1]
    rects = []
    class_id = 0
    for i in range(1, len(data) - 1):
        rect_str, class_id = data[i].split(',')[:4], data[i].split(',')[-1]
        rect = list(map(int, map(float, rect_str)))
        rect.append(int(class_id))
        rects.append(rect)

    return file_path, np.array(rects), image_orientation
