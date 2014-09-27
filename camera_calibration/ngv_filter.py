
import numpy as np
import ctypes

from multiprocessing import Pool, RawArray

from theano import *
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as cuda_ops

import cv2

import os
from gpu_camera_view import *

IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH = 1280, 720, 3
CROP_WIDTH, CROP_HEIGHT = 20, 20

FILTER_CONFIGURATION_FILE = os.path.abspath('filter_configurations.pkl')


class NGVStructure(object):
    __radian_order = [op.truediv(angle * math.pi, 180)
                      for angle in np.arange(0.0, 270.0, 0.5)]
    __total_radian = len(__radian_order)

    @staticmethod
    def create_ordered_structure(elements):
        total = NGVStructure.__total_radian
        structure = [[] for angle in xrange(total)]
        count = 0
        #
        # add elements to corresponding radian range
        for e in elements:
            if e['Type'] != 3:
                #
                # Do not add to structure if element type is 'Not Processed'
                # (see AppState.properties[CameraCalibration][Types])
                sigma = op.truediv(e['RelativeDegree'] * math.pi, 180)
                for j in xrange(0, total):
                    if sigma <= NGVStructure.__radian_order[j]:
                        structure[j].append([e['RealWorldDistance'], count])
                        break
            count += 1
        #
        # order elements by distance from origin
        for index in xrange(total):
            structure[index] = sorted(structure[index], key=lambda l: l[0])
        return structure

    @staticmethod
    def save_ordered_structure_to_pickle_file(file_path, elements):
        s = NGVStructure.create_ordered_structure(elements)

        if os.path.exists(file_path):
            os.remove(file_path)

        print 'saving calibration structure to pickle file... ({})'.format(
            file_path)

        pkl_file = open(file_path, 'wb')
        cPickle.dump(s, pkl_file, -1)
        pkl_file.close()


class UtilityOperations(object):

    @staticmethod
    def elements_to_image(elements):
        img = np.hstack(elements[0:IMAGE_WIDTH / CROP_WIDTH])
        for i in xrange(1, IMAGE_HEIGHT / CROP_HEIGHT):
            s = i * (IMAGE_WIDTH / CROP_WIDTH)
            e = s + (IMAGE_WIDTH / CROP_WIDTH)
            img = np.vstack((img, np.hstack(elements[s:e])))
        return img

    @staticmethod
    def crop_to_720p(img):
        if img.shape == (734, 1292, 3):
            return img[14:, 5:1285]
        # image height = img.shape[0]
        # image width = img.shape[1]
        if img.shape[0] > 720 or img.shape[1] > 1280:
            img_h, img_w = img.shape[0], img.shape[1]
            s_h, s_w_1, s_w_2 = 0, 0, 1280
            if img_h > 720:
                s_h = img_h - 720
            if img_w > 1280:
                s_w = img_w - 1280
                s_w_1 = (s_w / 2)
                s_w_2 = img_w - s_w_1
            return img[s_h:, (s_w_1 - 1):(s_w_2 - 1)]
        return img

    @staticmethod
    def convert_np_image_to_qt_image(image):
        image = UtilityOperations.crop_to_720p(image)
        return np.reshape(np.uint32(TheanoOperations.convert_to_qt_pixels(
            image)), (720, 1280))

    @staticmethod
    def convert_np_image_to_qt_elements(image):
        image = UtilityOperations.crop_to_720p(image)
        return np.uint32(TheanoOperations.convert_to_qt_elements(image))

    @staticmethod
    def qt_elements_classification(image):
        image = cv2.cvtColor(UtilityOperations.crop_to_720p(image),
                             cv2.COLOR_BGR2RGB)
        return TheanoOperations.apply_qt_elements_filtering(image)

    @staticmethod
    def qt_elements_filtering(image):
        image = cv2.cvtColor(UtilityOperations.crop_to_720p(image),
                             cv2.COLOR_BGR2RGB)
        return TheanoOperations.apply_qt_filtering(image)

    @staticmethod
    def qt_image_filtering(image):
        image = cv2.cvtColor(UtilityOperations.crop_to_720p(image),
                             cv2.COLOR_BGR2RGB)
        r = UtilityOperations.elements_to_image(
            TheanoOperations.apply_qt_filtering(image))
        return r


class IndexGenerator(object):

    @staticmethod
    def generate_crop_indexes_3d():
        indexes_3d = []
        for row in xrange(0, IMAGE_HEIGHT, CROP_HEIGHT):
            for col in xrange(0, IMAGE_WIDTH / CROP_WIDTH):
                indexes = []
                for c in xrange(CROP_HEIGHT):
                    indexes.append(range(c * IMAGE_WIDTH,
                                         c * IMAGE_WIDTH + CROP_HEIGHT))
                indexes_3d.append(np.add(indexes,
                                         col * CROP_WIDTH + row * IMAGE_WIDTH))
        return np.asarray(indexes_3d, dtype=np.int64)

    @staticmethod
    def generate_crop_pixels_indexes_3d():
        def channel_indexes(pixel_coord):
            channel_coord = long(pixel_coord) * 3
            return range(channel_coord, channel_coord + 3)
        crop_indexes = IndexGenerator.generate_crop_indexes_3d()
        pixel_channel_indexes = np.zeros(crop_indexes.shape + (3,),
                                         dtype=np.int64)
        for i in xrange(len(crop_indexes)):
            for j in xrange(len(crop_indexes[0])):
                for k in xrange(len(crop_indexes[0][0])):
                    pixel_channel_indexes[i][j][k] = \
                        channel_indexes(crop_indexes[i][j][k])
        return np.int64(pixel_channel_indexes)

    @staticmethod
    def red_channel_pixels_index():
        return IndexGenerator.generate_crop_indexes_3d() * 3

    @staticmethod
    def green_channel_pixels_index():
        return np.asarray(IndexGenerator.generate_crop_indexes_3d() * 3 + 1)

    @staticmethod
    def blue_channel_pixels_index():
        return np.asarray(IndexGenerator.generate_crop_indexes_3d() * 3 + 2)


#
# worker thread GLOBAL functions
# (do not encapsulate in a class)
def global_image_channel_cropping(channel_id):
    """
    worker thread function
    purpose:
        multiprocessing channel splitting and image cropping
    """
    if channel_id == 0:     # RED channel
        np.copyto(g_np_red_channel, g_np_image.take(g_np_red_indexes))

    elif channel_id == 1:   # GREEN channel
        np.copyto(g_np_green_channel, g_np_image.take(g_np_green_indexes))

    elif channel_id == 2:   # BLUE channel
        np.copyto(g_np_blue_channel, g_np_image.take(g_np_blue_indexes))


def global_elements_threshold_classification(index):
    count = np.count_nonzero(g_np_filtered[index])
    if count < g_np_threshold[0]:
        g_np_classified_elements[index] = 0
    else:
        g_np_classified_elements[index] = count


def global_image_threshold_classification(index):

    #if np.count_nonzero(np_filtered[index]) <= np_threshold[0]:
    #    np_filtered[index].fill(0)
    #else:
    #    np_filtered[index].fill(255)

    count = np.count_nonzero(g_np_filtered[index])

    #if count > 0:
    #    print count

    if count > 255:
        g_np_filtered[index].fill(255)
    elif count <= g_np_threshold[0]:
        g_np_filtered[index].fill(0)
    else:
        g_np_filtered[index].fill(count)

    #if np.count_nonzero(np_filtered[index]) <= np_threshold[0]:
    #    np.copyto(np_filtered[index], np_empty_crop)


qt_gray = [QColor(i, i, i) for i in xrange(256)]


def global_qt_image_threshold_classification(index):
    # black pixel = 4278190080
    # white pixel = 4294967295
    count = np.count_nonzero(g_np_filtered[index])
    if count < g_np_threshold[0]:
        g_np_qt_threshold[index].fill(4278190080)
    else:
        g_np_qt_threshold[index].fill(4294967295)


def global_pool_initializer(raw_array_image, image_shape,
                            raw_array_red_indexes, red_indexes_shape,
                            raw_array_green_indexes, green_indexes_shape,
                            raw_array_blue_indexes, blue_indexes_shape,
                            raw_array_red_channel, red_channel_shape,
                            raw_array_green_channel, green_channel_shape,
                            raw_array_blue_channel, blue_channel_shape,
                            raw_array_threshold, raw_array_filtered,
                            filtered_shape, raw_array_classified_elements,
                            classified_elements_shape, raw_array_qt_filtered,
                            qt_filtered_shape):
        global g_np_image
        g_np_image = np.ndarray(shape=image_shape, buffer=raw_array_image,
                                dtype=ctypes.c_uint8)
        global g_np_red_indexes
        g_np_red_indexes = np.ndarray(shape=red_indexes_shape,
                                      buffer=raw_array_red_indexes,
                                      dtype=ctypes.c_int32)
        global g_np_green_indexes
        g_np_green_indexes = np.ndarray(shape=green_indexes_shape,
                                        buffer=raw_array_green_indexes,
                                        dtype=ctypes.c_int32)
        global g_np_blue_indexes
        g_np_blue_indexes = np.ndarray(shape=blue_indexes_shape,
                                       buffer=raw_array_blue_indexes,
                                       dtype=ctypes.c_int32)
        global g_np_red_channel
        g_np_red_channel = np.ndarray(shape=red_channel_shape,
                                      buffer=raw_array_red_channel,
                                      dtype=ctypes.c_uint8)
        global g_np_green_channel
        g_np_green_channel = np.ndarray(shape=green_channel_shape,
                                        buffer=raw_array_green_channel,
                                        dtype=ctypes.c_uint8)
        global g_np_blue_channel
        g_np_blue_channel = np.ndarray(shape=blue_channel_shape,
                                       buffer=raw_array_blue_channel,
                                       dtype=ctypes.c_uint8)
        global g_np_filtered
        g_np_filtered = np.ndarray(shape=filtered_shape,
                                   buffer=raw_array_filtered,
                                   dtype=ctypes.c_float)
        global g_np_threshold
        g_np_threshold = np.ndarray(shape=(1,),
                                    buffer=raw_array_threshold,
                                    dtype=ctypes.c_float)
        global g_np_qt_threshold
        g_np_qt_threshold = np.ndarray(shape=qt_filtered_shape,
                                       buffer=raw_array_qt_filtered,
                                       dtype=ctypes.c_uint32)
        global g_np_classified_elements
        g_np_classified_elements = np.ndarray(
            shape=classified_elements_shape,
            buffer=raw_array_classified_elements, dtype=ctypes.c_uint8)
        global g_np_empty_crop
        g_np_empty_crop = np.zeros((20, 20), dtype=ctypes.c_float)


class MultiprocessOperations(object):
    __total_elements__ = range((IMAGE_WIDTH * IMAGE_HEIGHT)
                               / (CROP_WIDTH * CROP_HEIGHT))
    #
    # initialize RED channel indexes
    # as numpy.ndarray in shared memory
    __red_indexes = IndexGenerator.red_channel_pixels_index()
    __raw_red_indexes = RawArray(ctypes.c_int32, __red_indexes.size)
    __np_red_indexes = np.ndarray(shape=__red_indexes.shape,
                                  buffer=__raw_red_indexes,
                                  dtype=ctypes.c_int32)
    np.copyto(__np_red_indexes, __red_indexes)
    assert __raw_red_indexes is __np_red_indexes.base
    #
    # initialize GREEN channel indexes
    # as numpy.ndarray in shared memory
    __green_indexes = IndexGenerator.green_channel_pixels_index()
    __raw_green_indexes = RawArray(ctypes.c_int32, __green_indexes.size)
    __np_green_indexes = np.ndarray(shape=__green_indexes.shape,
                                    buffer=__raw_green_indexes,
                                    dtype=ctypes.c_int32)
    np.copyto(__np_green_indexes, __green_indexes)
    assert __raw_green_indexes is __np_green_indexes.base
    #
    # initialize BLUE channel indexes
    # as numpy.ndarray in shared memory
    __blue_indexes = IndexGenerator.blue_channel_pixels_index()
    __raw_blue_indexes = RawArray(ctypes.c_int32, __blue_indexes.size)
    __np_blue_indexes = np.ndarray(shape=__blue_indexes.shape,
                                   buffer=__raw_blue_indexes,
                                   dtype=ctypes.c_int32)
    np.copyto(__np_blue_indexes, __blue_indexes)
    assert __raw_blue_indexes is __np_blue_indexes.base
    #
    # initialize RED channel
    # as numpy.ndarray in shared memory
    __raw_red_channel = RawArray(ctypes.c_uint8, __red_indexes.size)
    __np_red_channel = np.ndarray(shape=__red_indexes.shape,
                                  buffer=__raw_red_channel,
                                  dtype=ctypes.c_uint8)
    assert __raw_red_channel is __np_red_channel.base
    # initialize GREEN channel
    # as numpy.ndarray in shared memory
    __raw_green_channel = RawArray(ctypes.c_uint8, __green_indexes.size)
    __np_green_channel = np.ndarray(shape=__green_indexes.shape,
                                    buffer=__raw_green_channel,
                                    dtype=ctypes.c_uint8)
    assert __raw_green_channel is __np_green_channel.base
    #
    # initialize BLUE channel
    # as numpy.ndarray in shared memory
    __raw_blue_channel = RawArray(ctypes.c_uint8, __blue_indexes.size)
    __np_blue_channel = np.ndarray(shape=__green_indexes.shape,
                                   buffer=__raw_blue_channel,
                                   dtype=ctypes.c_uint8)
    assert __raw_blue_channel is __np_blue_channel.base
    #
    # initialize IMAGE
    # as numpy.ndarray in shared memory
    __image_zeros = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH),
                             dtype=np.uint8)
    __raw_image = RawArray(ctypes.c_uint8, __image_zeros.size)
    __np_image = np.ndarray(shape=__image_zeros.shape, buffer=__raw_image,
                            dtype=ctypes.c_uint8)
    assert __raw_image is __np_image.base
    #
    # initialize FILTERED image
    # as numpy.ndarray in shared memory
    __raw_filtered = RawArray(ctypes.c_float, __np_red_indexes.size)
    __np_filtered = np.ndarray(shape=__np_red_indexes.shape,
                               buffer=__raw_filtered, dtype=ctypes.c_float)
    assert __raw_filtered is __np_filtered.base
    #
    # initialize QT FILTERED image
    __raw_qt_filtered = RawArray(ctypes.c_uint32, __np_red_indexes.size)
    __np_qt_filtered = np.ndarray(shape=__np_red_indexes.shape,
                                  buffer=__raw_qt_filtered,
                                  dtype=ctypes.c_uint32)
    assert __raw_qt_filtered is __np_qt_filtered.base
    #
    # initialize THRESHOLD INDEXES
    # as numpy.ndarray in shared memory
    __raw_threshold = RawArray(ctypes.c_float, 1)
    __np_threshold = np.ndarray(shape=(1,), buffer=__raw_threshold,
                                dtype=ctypes.c_float)
    assert __raw_threshold is __np_threshold.base
    #
    # initialize CLASSIFIED ELEMENTS
    # as numpy.ndarray in shared memory
    __raw_classified_elements = RawArray(ctypes.c_uint8,
                                         len(__total_elements__))
    __np_classified_elements = np.ndarray(shape=(len(__total_elements__), ),
                                          buffer=__raw_classified_elements,
                                          dtype=ctypes.c_uint8)
    assert __raw_classified_elements is __np_classified_elements.base
    #
    # initialize multiprocessing POOL
    # with initializer function for each processes
    __pool__ = Pool(
        processes=4, initializer=global_pool_initializer,
        initargs=(
            __raw_image, __np_image.shape,
            __raw_red_indexes, __np_red_indexes.shape,
            __raw_green_indexes, __np_green_indexes.shape,
            __raw_blue_indexes, __np_blue_indexes.shape,
            __raw_red_channel, __np_red_channel.shape,
            __raw_green_channel, __np_green_channel.shape,
            __raw_blue_channel, __np_blue_channel.shape,
            __np_threshold, __np_filtered, __np_filtered.shape,
            __np_classified_elements, __np_classified_elements.shape,
            __raw_qt_filtered, __np_qt_filtered.shape))

    shared_memory_red_channel = __np_red_channel
    shared_memory_green_channel = __np_green_channel
    shared_memory_blue_channel = __np_blue_channel

    shared_memory_image = __np_image
    shared_memory_filtered_elements = __np_filtered
    shared_memory_classified_elements = __np_classified_elements
    shared_memory_qt_filtered_elements = __np_qt_filtered

    @staticmethod
    def wait_for_element_cropping():
        MultiprocessOperations.__pool__.map_async(
            func=global_image_channel_cropping,
            iterable=[0, 1, 2]).get()

    @staticmethod
    def wait_for_elements_classification():
        MultiprocessOperations.__pool__.map_async(
            func=global_elements_threshold_classification,
            iterable=MultiprocessOperations.__total_elements__).get()

    @staticmethod
    def wait_for_image_classification():
        MultiprocessOperations.__pool__.map_async(
            func=global_image_threshold_classification,
            iterable=MultiprocessOperations.__total_elements__).get()

    @staticmethod
    def wait_for_qt_image_classification():
        MultiprocessOperations.__pool__.map_async(
            func=global_qt_image_threshold_classification,
            iterable=MultiprocessOperations.__total_elements__).get()

    @staticmethod
    def update_white_pixels_threshold(percent_value):
        MultiprocessOperations.__np_threshold[0] = \
            percent_value * CROP_HEIGHT * CROP_WIDTH

    current_threshold = __np_threshold[0]


#
# Theano GPU GLOBAL functions
# (do not encapsulate in a class)
def global_image_classification(red, green, blue, params):
    # __shared_params[0] = min_overall
    # __shared_params[1] = min_blue
    # __shared_params[2] = max_blue
    # __shared_params[3] = min_blue_green_diff
    # __shared_params[4] = max_blue_green_diff
    # __shared_params[5] = min_green_red_diff
    # __shared_params[6] = max_green_red_diff
    # [T.lt(a, b)] <--> [a < b]
    bg_diff = blue - green
    gr_diff = green - red
    return T.cast(
        T.lt(params[1], blue) &
        T.lt(blue, params[2]) &
        T.lt(params[0], red) &
        T.lt(params[0], green) &
        T.lt(params[3], bg_diff) &
        T.lt(bg_diff, params[4]) &
        T.lt(params[5], gr_diff) &
        T.lt(gr_diff, params[6]),
        'float32')


def global_qt_image_filtering(red, green, blue, params):
    # shared_params[0] = min_overall
    # shared_params[1] = min_blue
    # shared_params[2] = max_blue
    # shared_params[3] = min_blue_green_diff
    # shared_params[4] = max_blue_green_diff
    # shared_params[5] = min_green_red_diff
    # shared_params[6] = max_green_red_diff
    # [T.lt(a, b)] <--> [a < b]
    bg_diff = blue - green
    gr_diff = green - red
    return (
        T.lt(params[1], blue) &
        T.lt(blue, params[2]) &
        T.lt(params[0], red) &
        T.lt(params[0], green) &
        T.lt(params[3], bg_diff) &
        T.lt(bg_diff, params[4]) &
        T.lt(params[5], gr_diff) &
        T.lt(gr_diff, params[6])
    ) * 16777215.0


def global_image_filtering(red, green, blue, params):
    # shared_params[0] = min_overall
    # shared_params[1] = min_blue
    # shared_params[2] = max_blue
    # shared_params[3] = min_blue_green_diff
    # shared_params[4] = max_blue_green_diff
    # shared_params[5] = min_green_red_diff
    # shared_params[6] = max_green_red_diff
    # [T.lt(a, b)] <--> [a < b]
    bg_diff = blue - green
    gr_diff = green - red
    return (
        T.lt(params[1], blue) &
        T.lt(blue, params[2]) &
        T.lt(params[0], red) &
        T.lt(params[0], green) &
        T.lt(params[3], bg_diff) &
        T.lt(bg_diff, params[4]) &
        T.lt(params[5], gr_diff) &
        T.lt(gr_diff, params[6])
    ) * 255.0


def global_convert_to_qt_pixel(red, green, blue):
    # QT uses [Alpha Blue Green Red] based colors
    # Theano doesn't support bit-shifting yet
    # alpha = 255 -> 255 * (2 ** 24) = 4 278 190 080
    # blue -> b * (2 ** 16) = b * 65 536
    # green -> g * (2 ** 8) = g * 256
    return 4278190080 + (blue * 65536) + (green * 256) + red


class TheanoOperations(object):
    #
    # element crops INDEXES
    __indexes__ = IndexGenerator.generate_crop_indexes_3d()
    __red_channel_indexes__ = IndexGenerator.red_channel_pixels_index()
    __green_channel_indexes__ = IndexGenerator.green_channel_pixels_index()
    __blue_channel_indexes__ = IndexGenerator.blue_channel_pixels_index()
    #
    # initialize image CHANNELS matrices (2 dimensions)
    # as Theano GPU shared variables
    __shared_red = shared(np.zeros_like(__red_channel_indexes__,
                                        dtype=config.floatX), borrow=True)
    __shared_green = shared(np.zeros_like(__green_channel_indexes__,
                                          dtype=config.floatX), borrow=True)
    __shared_blue = shared(np.zeros_like(__blue_channel_indexes__,
                                         dtype=config.floatX), borrow=True)
    #
    # initialize image CHANNELS vector (1 dimensions)
    # as Theano GPU shared variables
    __shared_flat_red = shared(np.zeros((IMAGE_WIDTH * IMAGE_HEIGHT),
                               dtype=np.uint32), borrow=True)
    __shared_flat_green = shared(np.zeros((IMAGE_WIDTH * IMAGE_HEIGHT),
                                 dtype=np.uint32), borrow=True)
    __shared_flat_blue = shared(np.zeros((IMAGE_WIDTH * IMAGE_HEIGHT),
                                dtype=np.uint32), borrow=True)
    #
    # initialize binary filtering PARAMETERS
    # as Theano GPU shared variable
    __shared_params = shared(np.zeros(7, dtype=config.floatX), borrow=True)
    #
    # initialize image binary CLASSIFICATION
    # as Theano variables
    __classification_results, __classification_updates = scan(
        lambda r, g, b, params: global_image_classification(r, g, b, params),
        sequences=[__shared_red, __shared_green, __shared_blue],
        non_sequences=[__shared_params])
    __apply_classification = function(
        inputs=[],
        outputs=Out(cuda_ops.gpu_from_host(__classification_results),
                    borrow=True))
    #
    # initialize image binary FILTERING
    # as Theano variables
    __filtering_results, __filtering_updates = scan(
        lambda r, g, b, params: global_image_filtering(r, g, b, params),
        sequences=[__shared_red, __shared_green, __shared_blue],
        non_sequences=[__shared_params])
    __apply_binary_filtering = function(
        inputs=[],
        outputs=Out(cuda_ops.gpu_from_host(__filtering_results),
                    borrow=True))
    #
    # initialize QT image binary FILTERING
    # as Theano variables
    __qt_filtering_results, __qt_filtering_updates = scan(
        lambda r, g, b, params: global_qt_image_filtering(r, g, b, params),
        sequences=[__shared_red, __shared_green, __shared_blue],
        non_sequences=[__shared_params])
    __qt_apply_binary_filtering = function(
        inputs=[],
        outputs=Out(cuda_ops.gpu_from_host(__qt_filtering_results),
                    borrow=True))
    #
    # initialize QT pixel colors CONVERSION
    # as Theano variables
    __convert_to_qt_results, __convert_to_qt_updates = scan(
        lambda r, g, b: global_convert_to_qt_pixel(r, g, b),
        sequences=[__shared_flat_red, __shared_flat_green, __shared_flat_blue])
    __convert_to_qt_image = function(
        inputs=[], outputs=Out(__convert_to_qt_results, borrow=True))
    #
    #

    @staticmethod
    def update_filtering_parameters(threshold, min_channel_value,
                                    min_blue, max_blue,
                                    min_blue_green_diff, max_blue_green_diff,
                                    min_green_red_diff, max_green_red_diff):
        """
        :type threshold: float
        :param threshold: percent of white pixels in crop element [0.0 to 1.0]

        :type min_channel_value: int
        :param min_channel_value: minimum value for any RGB channels [0 to 255]

        :type min_blue: int
        :param min_blue: minimum value for BLUE channel [0 to 255]

        :type max_blue: int
        :param max_blue: maximum value for BLUE channel [0 to 255]

        :type min_blue_green_diff: int
        :param min_blue_green_diff: minimum difference for BLUE and GREEN
                                    channels (BLUE - GREEN) [0 to 255]

        :type max_blue_green_diff: int
        :param max_blue_green_diff: maximum difference for BLUE and GREEN
                                    channels (BLUE - GREEN) [0 to 255]

        :type min_green_red_diff: int
        :param min_green_red_diff: minimum difference for GREEN and RED
                                    channels (GREEN - RED) [0 to 255]

        :type max_green_red_diff: int
        :param max_green_red_diff: maximum difference for GREEN and RED
                                    channels (GREEN - RED) [0 to 255]
        """
        MultiprocessOperations.update_white_pixels_threshold(threshold)

        TheanoOperations.__shared_params.set_value([
            min_channel_value, min_blue, max_blue, min_blue_green_diff,
            max_blue_green_diff, min_green_red_diff, max_green_red_diff],
            borrow=True)

    current_parameters = __shared_params.get_value()

    @staticmethod
    def apply_qt_filtering(image):
        #
        # apply bilateral filter on IMAGE to smooth colors while keeping edges
        cv2.bilateralFilter(UtilityOperations.crop_to_720p(image), 3, 255, 50,
                            dst=MultiprocessOperations.shared_memory_image)
        #
        # crop elements from IMAGE
        MultiprocessOperations.wait_for_element_cropping()
        #
        # upload RED channels to GPU
        TheanoOperations.__shared_red.set_value(
            MultiprocessOperations.shared_memory_red_channel, borrow=True)
        #
        # upload GREEN channels to GPU
        TheanoOperations.__shared_green.set_value(
            MultiprocessOperations.shared_memory_green_channel, borrow=True)
        #
        # upload BLUE channels to GPU
        TheanoOperations.__shared_blue.set_value(
            MultiprocessOperations.shared_memory_blue_channel, borrow=True)
        return np.uint32(np.asarray(
            TheanoOperations.__qt_apply_binary_filtering()))

    @staticmethod
    def apply_filtering(image):
        #
        # apply bilateral filter on IMAGE to smooth colors while keeping edges
        cv2.bilateralFilter(UtilityOperations.crop_to_720p(image), 3, 255, 50,
                            dst=MultiprocessOperations.shared_memory_image)
        #
        # crop elements from IMAGE
        MultiprocessOperations.wait_for_element_cropping()
        #
        # upload RED channels to GPU
        TheanoOperations.__shared_red.set_value(
            MultiprocessOperations.shared_memory_red_channel, borrow=True)
        #
        # upload GREEN channels to GPU
        TheanoOperations.__shared_green.set_value(
            MultiprocessOperations.shared_memory_green_channel, borrow=True)
        #
        # upload BLUE channels to GPU
        TheanoOperations.__shared_blue.set_value(
            MultiprocessOperations.shared_memory_blue_channel, borrow=True)
        #
        # download FILTERING result from GPU
        np.copyto(MultiprocessOperations.shared_memory_filtered_elements,
                  TheanoOperations.__apply_binary_filtering())
        #
        # apply IMAGE threshold CLASSIFICATION
        MultiprocessOperations.wait_for_image_classification()
        return MultiprocessOperations.shared_memory_filtered_elements

    @staticmethod
    def apply_elements_classification(image):
        #
        # apply bilateral filter on IMAGE to smooth colors while keeping edges
        cv2.bilateralFilter(UtilityOperations.crop_to_720p(image), 3, 255, 50,
                            dst=MultiprocessOperations.shared_memory_image)
        #
        # crop elements from IMAGE
        MultiprocessOperations.wait_for_element_cropping()
        #
        # upload RED channels to GPU
        TheanoOperations.__shared_red.set_value(
            MultiprocessOperations.shared_memory_red_channel, borrow=True)
        #
        # upload GREEN channels to GPU
        TheanoOperations.__shared_green.set_value(
            MultiprocessOperations.shared_memory_green_channel, borrow=True)
        #
        # upload BLUE channels to GPU
        TheanoOperations.__shared_blue.set_value(
            MultiprocessOperations.shared_memory_blue_channel, borrow=True)
        #
        # download FILTERING result from GPU
        np.copyto(MultiprocessOperations.shared_memory_filtered_elements,
                  TheanoOperations.__apply_classification())
        #
        # apply ELEMENTS threshold CLASSIFICATION
        MultiprocessOperations.wait_for_elements_classification()
        return MultiprocessOperations.shared_memory_classified_elements

    @staticmethod
    def apply_qt_elements_filtering(image):
        #
        # apply bilateral filter on IMAGE to smooth colors while keeping edges
        cv2.bilateralFilter(UtilityOperations.crop_to_720p(image), 3, 255, 50,
                            dst=MultiprocessOperations.shared_memory_image)
        #
        # crop elements from IMAGE
        MultiprocessOperations.wait_for_element_cropping()

        #
        # upload RED channels to GPU
        TheanoOperations.__shared_red.set_value(
            MultiprocessOperations.shared_memory_red_channel, borrow=True)
        #
        # upload GREEN channels to GPU
        TheanoOperations.__shared_green.set_value(
            MultiprocessOperations.shared_memory_green_channel, borrow=True)
        #
        # upload BLUE channels to GPU
        TheanoOperations.__shared_blue.set_value(
            MultiprocessOperations.shared_memory_blue_channel, borrow=True)
        #
        # download FILTERING result from GPU
        np.copyto(MultiprocessOperations.shared_memory_filtered_elements,
                  TheanoOperations.__apply_binary_filtering())
        #
        # apply IMAGE threshold CLASSIFICATION
        MultiprocessOperations.wait_for_qt_image_classification()
        return MultiprocessOperations.shared_memory_qt_filtered_elements

    @staticmethod
    def convert_to_qt_pixels(image):
        r, g, b = cv2.split(image)
        TheanoOperations.__shared_flat_red.set_value(r.flatten(), borrow=True)
        TheanoOperations.__shared_flat_green.set_value(g.flatten(), borrow=True)
        TheanoOperations.__shared_flat_blue.set_value(b.flatten(), borrow=True)
        return TheanoOperations.__convert_to_qt_image()

    @staticmethod
    def convert_to_qt_elements(image):
        qt_image = TheanoOperations.convert_to_qt_pixels(image)
        return np.take(qt_image, TheanoOperations.__indexes__)


class NGVFilter(object):
    """""
    le reste est encapsuler dans cette element

    pour le array de mapping (la distance et
    l'angle par rapport a l'origine) dit moi qu'est-ce que tu veux
    comme structure par email et si tu veux que je te que
    les mettent en ordre par angle ou par distance ou par ID

    ex: list = [{distance: x, angle_radian: y}] x 2304 elements



    Start GUI
    Filter Image
    Save Element Structure
    Load and Save Filter parameters
    Index dans laffichage de Zoomed Element

    """""

    @staticmethod
    def start_calibration_gui(camera_object):
        """ demarre le gui de calibration gpu_camera_view avec un objet
            Camera de CameraLib
        """
        start_calibration(camera_object)

    @staticmethod
    def load_filter_configurations_from_file(file_path):
        """  load filter configurations from pickle file"""
        if not os.path.exists(file_path):
            print 'ERROR: Unable to load filter configuration file ({})'.format(
                FILTER_CONFIGURATION_FILE)
            return

        pkl_file = open(file_path, 'rb')
        configuration = cPickle.load(pkl_file)
        pkl_file.close()

        TheanoOperations.update_filtering_parameters(
            configuration['Threshold'], configuration['MinValue'],
            configuration['MinBlue'], configuration['MaxBlue'],
            configuration['MinBlueGreenDiff'],
            configuration['MaxBlueGreenDiff'],
            configuration['MinGreenRedDiff'], configuration['MaxGreenRedDiff']
        )

    @staticmethod
    def save_filter_configurations_to_file(file_path):
        """ save filter configurations to pickle to pickle file"""
        if os.path.exists(file_path):
            os.remove(file_path)

        configuration = {
            'Threshold': MultiprocessOperations.current_threshold,
            'MinValue': TheanoOperations.current_parameters[0],
            'MinBlue': TheanoOperations.current_parameters[1],
            'MaxBlue': TheanoOperations.current_parameters[2],
            'MinBlueGreenDiff': TheanoOperations.current_parameters[3],
            'MaxBlueGreenDiff': TheanoOperations.current_parameters[4],
            'MinGreenRedDiff': TheanoOperations.current_parameters[5],
            'MaxGreenRedDiff': TheanoOperations.current_parameters[6]}
        pkl_file = open(file_path, 'wb')
        cPickle.dump(configuration, pkl_file, -1)
        pkl_file.close()

    @staticmethod
    def save_elements_structure(file_path, elements):
        NGVStructure.save_ordered_structure_to_pickle_file(
            file_path, elements)
    __loaded = False

    @staticmethod
    def update_configurations(
            classification_threshold, min_channel_value, min_blue_value,
            max_blue_value, min_blue_green_diff, max_blue_green_diff,
            min_green_red_diff, max_green_red_diff):
        """ update filter configurations """
        if not NGVFilter.__loaded:
            NGVFilter.__loaded = True

        TheanoOperations.update_filtering_parameters(
            classification_threshold, min_channel_value,  min_blue_value,
            max_blue_value, min_blue_green_diff, max_blue_green_diff,
            min_green_red_diff, max_green_red_diff)

    @staticmethod
    def elements_weighted_classification(numpy_image):
        """ retourne un array de 1 dimension contenant 2304 crop d'image
            avec le nombre de pixels blanc dans chacune des cases
            ******(c'est cette function que tu appelles)*****

            USE RGB BASED IMAGES
        """
        if not NGVFilter.__loaded:
            NGVFilter.load_filter_configurations_from_file(
                FILTER_CONFIGURATION_FILE)
            NGVFilter.__loaded = True

        return TheanoOperations.apply_elements_classification(
            UtilityOperations.crop_to_720p(numpy_image))
