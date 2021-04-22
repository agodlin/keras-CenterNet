import numpy as np
import cv2
from PIL import Image
from functools import partial
from yolo.yolo import yolo_input_shape
def get_lut():
    convert = [0, 9, 17, 26, 31, 37, 42, 47, 50, 54, 57, 60, 63, 67, 70, 72, 74, 76, 78, 80, 82, 83, 85, 87, 89, 91, 93,
               94, 95, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 114, 115, 116, 117, 117, 118,
               119, 120, 120, 121, 122, 122, 123, 124, 125, 125, 126, 127, 128, 128, 129, 130, 130, 131, 132, 133, 133,
               134, 135, 135, 136, 137, 138, 138, 139, 139, 140, 140, 141, 141, 142, 142, 142, 143, 143, 144, 144, 145,
               145, 146, 146, 146, 147, 147, 148, 148, 149, 149, 149, 150, 150, 151, 151, 152, 152, 152, 153, 153, 154,
               154, 155, 155, 155, 156, 156, 157, 157, 158, 158, 159, 159, 159, 160, 160, 161, 161, 162, 162, 162, 163,
               163, 163, 163, 164, 164, 164, 164, 165, 165, 165, 165, 166, 166, 166, 166, 167, 167, 167, 167, 168, 168,
               168, 169, 169, 169, 169, 170, 170, 170, 170, 171, 171, 171, 171, 172, 172, 172, 172, 173, 173, 173, 173,
               174, 174, 174, 175, 175, 175, 175, 176, 176, 176, 176, 177, 177, 177, 177, 178, 178, 178, 178, 179, 179,
               179, 180, 180, 180, 180, 181, 181, 181, 181, 182, 182, 182, 182, 183, 183, 183, 183, 184, 184, 184, 184,
               185, 185, 185, 185, 185, 186, 186, 186, 186, 186, 186, 187, 187, 187, 187, 187, 187, 188, 188, 188, 188,
               188, 188, 188, 189, 189, 189, 189, 189, 189, 190, 190, 190, 190, 190, 190, 190, 191, 191, 191, 191, 191,
               191, 192, 192, 192, 192, 192, 192, 193, 193, 193, 193, 193, 193, 193, 194, 194, 194, 194, 194, 194, 195,
               195, 195, 195, 195, 195, 196, 196, 196, 196, 196, 196, 196, 197, 197, 197, 197, 197, 197, 198, 198, 198,
               198, 198, 198, 198, 199, 199, 199, 199, 199, 199, 200, 200, 200, 200, 200, 200, 201, 201, 201, 201, 201,
               201, 201, 202, 202, 202, 202, 202, 202, 203, 203, 203, 203, 203, 203, 204, 204, 204, 204, 204, 204, 204,
               205, 205, 205, 205, 205, 205, 206, 206, 206, 206, 206, 206, 206, 207, 207, 207, 207, 207, 207, 208, 208,
               208, 208, 208, 208, 209, 209, 209, 209, 209, 209, 209, 209, 209, 210, 210, 210, 210, 210, 210, 210, 210,
               210, 210, 210, 211, 211, 211, 211, 211, 211, 211, 211, 211, 211, 212, 212, 212, 212, 212, 212, 212, 212,
               212, 212, 212, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 213, 214, 214, 214, 214, 214, 214, 214,
               214, 214, 214, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 215, 216, 216, 216, 216, 216, 216, 216,
               216, 216, 216, 216, 217, 217, 217, 217, 217, 217, 217, 217, 217, 217, 218, 218, 218, 218, 218, 218, 218,
               218, 218, 218, 218, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 219, 220, 220, 220, 220, 220, 220,
               220, 220, 220, 220, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 222, 222, 222, 222, 222, 222,
               222, 222, 222, 222, 222, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 223, 224, 224, 224, 224, 224,
               224, 224, 224, 224, 224, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 226, 226, 226, 226, 226,
               226, 226, 226, 226, 226, 226, 227, 227, 227, 227, 227, 227, 227, 227, 227, 227, 228, 228, 228, 228, 228,
               228, 228, 228, 228, 228, 228, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 230, 230, 230, 230,
               230, 230, 230, 230, 230, 230, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 232, 232, 232, 232,
               232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233, 233,
               233, 233, 233, 233, 233, 233, 233, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234, 234,
               234, 234, 234, 234, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235, 235,
               236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 236, 237, 237, 237,
               237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 237, 238, 238, 238, 238, 238, 238,
               238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 238, 239, 239, 239, 239, 239, 239, 239, 239, 239, 239,
               239, 239, 239, 239, 239, 239, 239, 239, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
               240, 240, 240, 240, 240, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
               241, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 243, 243,
               243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 244, 244, 244, 244, 244,
               244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 244, 245, 245, 245, 245, 245, 245, 245, 245, 245,
               245, 245, 245, 245, 245, 245, 245, 245, 245, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246,
               246, 246, 246, 246, 246, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247,
               247, 247, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 248, 249,
               249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250,
               250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251, 251, 251, 251, 251,
               251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,
               252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
               253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
               255, 255, 255, 255, 255, 255, 255, 255, 255]
    return convert

def pixel_convert(x):
    x = min(x,1023)
    x = max(x, 0)
    convert = get_lut()

    return convert[x]


vf = np.vectorize(pixel_convert)

def lut_int8(image):
    shape = image.shape
    image = image.flatten()
    lut = np.array(get_lut())
    preprocess_image = lut[image]

    preprocess_image = preprocess_image.reshape(shape)
    preprocess_image = preprocess_image.astype(np.uint8)
    return preprocess_image


def lut_int8_old(image):
    shape = image.shape
    image = image.flatten()
    preprocess_image = vf(image)

    preprocess_image = preprocess_image.reshape(shape)
    preprocess_image = preprocess_image.astype(np.uint8)
    return preprocess_image


def letterbox_image_int(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw = image.shape[:2]
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx, dy = ((w-nw)//2, (h-nh)//2)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_NEAREST)
    new_image = Image.fromarray(np.zeros((w, h), dtype=image.dtype))
    new_image.paste(Image.fromarray(image), (dx, dy))
    image = np.asarray(new_image)
    image_data = np.expand_dims(image, axis=2)

    return image_data

def yolo_w10_image_preprocess(image, image_normalizer):
    boxed_image = letterbox_image_int(image, yolo_input_shape)

    boxed_image = lut_int8(boxed_image)

    return image_normalizer(boxed_image)

def yolo_test_w10_image_preprocess(image, image_normalizer):
    boxed_image = letterbox_image_int(image, yolo_input_shape)

    return image_normalizer(boxed_image)


def yolo_gray_image_preprocess(image, image_normalizer):
    boxed_image = letterbox_image_int(image, yolo_input_shape)

    return image_normalizer(boxed_image)


def yolo_image_normalization_flaot(image):
    boxed_image = image.astype(np.float32)
    boxed_image = (boxed_image / 255) - 0.5
    boxed_image[boxed_image > 0.5] = 0.5
    boxed_image[boxed_image < -0.5] = -0.5
    return boxed_image

def yolo_1023_image_normalization_flaot(image):
    boxed_image = image.astype(np.float32)
    boxed_image = (boxed_image / 1023) - 0.5
    boxed_image[boxed_image > 0.5] = 0.5
    boxed_image[boxed_image < -0.5] = -0.5
    return boxed_image

def yolo_image_normalization_int(image):
    boxed_image = image.astype(np.int32)
    boxed_image = boxed_image - 128
    boxed_image = boxed_image.astype(np.int8)
    return boxed_image

def yolo_movidius_image_preprocess(image, image_normalizer):
    from preprocessing import w10_movidius_preprocess
    boxed_image = letterbox_image_int(image, yolo_input_shape)
    boxed_image = w10_movidius_preprocess(boxed_image)
    return image_normalizer(boxed_image)

def identity(image):
    return image

def image_preprocess_factory(type):
    if type == 'w10_int':
        return partial(yolo_w10_image_preprocess, image_normalizer=yolo_image_normalization_int)
    if type == 'w10_float':
        return partial(yolo_w10_image_preprocess, image_normalizer=yolo_image_normalization_flaot)
    if type == 'gray_int':
        return partial(yolo_gray_image_preprocess, image_normalizer=yolo_image_normalization_int)
    if type == 'gray_float':
        return partial(yolo_gray_image_preprocess, image_normalizer=yolo_image_normalization_flaot)
    if type == 'test_float':
        return partial(yolo_test_w10_image_preprocess, image_normalizer=yolo_1023_image_normalization_flaot)
    if type == 'movidius':
        return partial(yolo_movidius_image_preprocess, image_normalizer=yolo_image_normalization_flaot)
    if type == 'none':
        return partial(identity)

