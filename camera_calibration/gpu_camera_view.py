
import timeit
import time
import ctypes
import cv2
import numpy as np
import camera

from multiprocessing import RawArray, Pool
from theano import *
import theano.tensor as T
import theano.sandbox.cuda.basic_ops as cuda_ops


from PySide.QtGui import *
from PySide.QtCore import QTimer, SIGNAL
import sys


def generate_crop_indexes_3d(width, height, crop_width, crop_height):
    idxs = []
    for row in xrange(0, height, crop_height):
        for col in xrange(0, width / crop_width):
            indexes = []
            for c in xrange(crop_height):
                indexes.append(range(c * width, c * width + crop_height))
            idxs.append(np.add(indexes, col * crop_width + row * width))
    return np.asarray(idxs, dtype=np.int64)


def generate_crop_pixels_indexes_3d(width, height, crop_width, crop_height):
    def channel_indexes(pixel_coord):
        channel_coord = long(pixel_coord) * 3
        return range(channel_coord, channel_coord + 3)

    crop_indexes = generate_crop_indexes_3d(width, height, crop_width, crop_height)
    pixel_channel_indexes = np.zeros(crop_indexes.shape + (3,),
                                     dtype=np.int64)

    for i in xrange(len(crop_indexes)):
        for j in xrange(len(crop_indexes[0])):
            for k in xrange(len(crop_indexes[0][0])):
                pixel_channel_indexes[i][j][k] = \
                    channel_indexes(crop_indexes[i][j][k])

    return np.int64(pixel_channel_indexes)


def red_channel_pixels_index(width, height, crop_width, crop_height):
    return generate_crop_indexes_3d(width, height, crop_width, crop_height) * 3


def green_channel_pixels_index(width, height, crop_width, crop_height):
    return np.asarray(generate_crop_indexes_3d(width, height, crop_width, crop_height) * 3 + 1)


def blue_channel_pixels_index(width, height, crop_width, crop_height):
    return np.asarray(generate_crop_indexes_3d(width, height, crop_width, crop_height) * 3 + 2)


def classification_by_threshold(index):

    #if np.count_nonzero(np_filtered[index]) <= np_threshold[0]:
    #    np_filtered[index].fill(0)
    #else:
    #    np_filtered[index].fill(255)

    count = np.count_nonzero(np_filtered[index])

    if count > 255:
        np_filtered[index].fill(255)
    elif count <= np_threshold[0]:
        np_filtered[index].fill(0)
    else:
        np_filtered[index].fill(count)

    #if np.count_nonzero(np_filtered[index]) <= np_threshold[0]:
    #    np.copyto(np_filtered[index], np_empty_crop)


def crop_image_channels(channel_id):
    """
    worker thread function
    purpose:
        multiprocessing channel splitting and image cropping
    """
    if channel_id == 0:     # RED channel
        np.copyto(np_red_channel, np_image.take(np_red_indexes))

    elif channel_id == 1:   # GREEN channel
        np.copyto(np_green_channel, np_image.take(np_green_indexes))

    elif channel_id == 2:   # BLUE channel
        np.copyto(np_blue_channel, np_image.take(np_blue_indexes))


class GpuNumpyQtOperations(object):
    def __init__(self, image_width, image_height, image_depth,
                 crop_width, crop_height):
        # initialize RED channel indexes
        # as numpy.ndarray in shared memory --------------------------------
        red_indexes = red_channel_pixels_index(image_width, image_height,
                                               crop_width, crop_height)
        raw_red_indexes = RawArray(ctypes.c_int32, red_indexes.size)
        np_red_indexes = np.ndarray(shape=red_indexes.shape,
                                    buffer=raw_red_indexes,
                                    dtype=ctypes.c_int32)
        np.copyto(np_red_indexes, red_indexes)
        assert raw_red_indexes is np_red_indexes.base
        # ------------------------------------------------------------------
        # initialize GREEN channel indexes
        # as numpy.ndarray in shared memory --------------------------------
        green_indexes = green_channel_pixels_index(image_width, image_height,
                                                   crop_width, crop_height)
        raw_green_indexes = RawArray(ctypes.c_int32, green_indexes.size)
        np_green_indexes = np.ndarray(shape=green_indexes.shape,
                                      buffer=raw_green_indexes,
                                      dtype=ctypes.c_int32)
        np.copyto(np_green_indexes, green_indexes)
        assert raw_green_indexes is np_green_indexes.base
        # ------------------------------------------------------------------
        # initialize BLUE channel indexes
        # as numpy.ndarray in shared memory --------------------------------
        blue_indexes = blue_channel_pixels_index(image_width, image_height,
                                                 crop_width, crop_height)
        raw_blue_indexes = RawArray(ctypes.c_int32, blue_indexes.size)
        np_blue_indexes = np.ndarray(shape=blue_indexes.shape,
                                     buffer=raw_blue_indexes,
                                     dtype=ctypes.c_int32)
        np.copyto(np_blue_indexes, blue_indexes)
        assert raw_blue_indexes is np_blue_indexes.base
        # ------------------------------------------------------------------
        # initialize RED channel
        # as numpy.ndarray in shared memory --------------------------------
        raw_red_channel = RawArray(ctypes.c_uint8, red_indexes.size)
        self.__np_red_channel = np.ndarray(shape=red_indexes.shape,
                                           buffer=raw_red_channel,
                                           dtype=ctypes.c_uint8)
        assert raw_red_channel is self.__np_red_channel.base
        # ------------------------------------------------------------------
        # initialize GREEN channel
        # as numpy.ndarray in shared memory --------------------------------
        raw_green_channel = RawArray(ctypes.c_uint8, green_indexes.size)
        self.__np_green_channel = np.ndarray(shape=green_indexes.shape,
                                             buffer=raw_green_channel,
                                             dtype=ctypes.c_uint8)
        assert raw_green_channel is self.__np_green_channel.base
        # ------------------------------------------------------------------
        # initialize BLUE channel
        # as numpy.ndarray in shared memory --------------------------------
        raw_blue_channel = RawArray(ctypes.c_uint8, blue_indexes.size)
        self.__np_blue_channel = np.ndarray(shape=green_indexes.shape,
                                            buffer=raw_blue_channel,
                                            dtype=ctypes.c_uint8)
        assert raw_blue_channel is self.__np_blue_channel.base
        # ------------------------------------------------------------------
        # initialize IMAGE
        # as numpy.ndarray in shared memory --------------------------------
        image_zeros = np.zeros((image_height, image_width,
                                image_depth)).astype(dtype=np.uint8)
        raw_image = RawArray(ctypes.c_uint8, image_zeros.size)
        self.__np_image = np.ndarray(shape=image_zeros.shape,
                                     buffer=raw_image,
                                     dtype=ctypes.c_uint8)
        assert raw_image is self.__np_image.base
        # ------------------------------------------------------------------
        # initialize FILTERED
        # as numpy.ndarray in shared memory --------------------------------
        raw_filtered = RawArray(ctypes.c_float, np_red_indexes.size)
        self.__np_filtered = np.ndarray(shape=np_red_indexes.shape,
                                        buffer=raw_filtered,
                                        dtype=ctypes.c_float)
        assert raw_filtered is self.__np_filtered.base
        # ------------------------------------------------------------------
        # initialize THRESHOLD
        # as numpy.ndarray in shared memory --------------------------------
        self.__threshold_args = range((image_width * image_height)
                                      / (crop_width * crop_height))
        raw_threshold = RawArray(ctypes.c_float, 1)
        self.__np_threshold = np.ndarray(shape=(1,),
                                         buffer=raw_threshold,
                                         dtype=ctypes.c_float)
        assert raw_threshold is self.__np_threshold.base
        # ------------------------------------------------------------------

        # initialize multiprocessing POOL
        # with initializer function for each processes ---------------------
        def init_pool_process(raw_array_image, image_shape,
                              raw_array_red_indexes, red_indexes_shape,
                              raw_array_green_indexes, green_indexes_shape,
                              raw_array_blue_indexes, blue_indexes_shape,
                              raw_array_red_channel, red_channel_shape,
                              raw_array_green_channel, green_channel_shape,
                              raw_array_blue_channel, blue_channel_shape,
                              raw_array_threshold, raw_array_filtered,
                              filtered_shape):
            global np_image
            np_image = np.ndarray(shape=image_shape,
                                  buffer=raw_array_image,
                                  dtype=ctypes.c_uint8)

            global np_red_indexes
            np_red_indexes = np.ndarray(shape=red_indexes_shape,
                                        buffer=raw_array_red_indexes,
                                        dtype=ctypes.c_int32)

            global np_green_indexes
            np_green_indexes = np.ndarray(shape=green_indexes_shape,
                                          buffer=raw_array_green_indexes,
                                          dtype=ctypes.c_int32)

            global np_blue_indexes
            np_blue_indexes = np.ndarray(shape=blue_indexes_shape,
                                         buffer=raw_array_blue_indexes,
                                         dtype=ctypes.c_int32)

            global np_red_channel
            np_red_channel = np.ndarray(shape=red_channel_shape,
                                        buffer=raw_array_red_channel,
                                        dtype=ctypes.c_uint8)

            global np_green_channel
            np_green_channel = np.ndarray(shape=green_channel_shape,
                                          buffer=raw_array_green_channel,
                                          dtype=ctypes.c_uint8)

            global np_blue_channel
            np_blue_channel = np.ndarray(shape=blue_channel_shape,
                                         buffer=raw_array_blue_channel,
                                         dtype=ctypes.c_uint8)

            global np_filtered
            np_filtered = np.ndarray(shape=filtered_shape,
                                     buffer=raw_array_filtered,
                                     dtype=ctypes.c_float)

            global np_threshold
            np_threshold = np.ndarray(shape=(1,),
                                      buffer=raw_array_threshold,
                                      dtype=ctypes.c_float)

            global np_empty_crop
            np_empty_crop = np.zeros((20, 20), dtype=ctypes.c_float)

        self.__pool = Pool(processes=6, initializer=init_pool_process,
                           initargs=(
                               raw_image, self.__np_image.shape,
                               raw_red_indexes, np_red_indexes.shape,
                               raw_green_indexes, np_green_indexes.shape,
                               raw_blue_indexes, np_blue_indexes.shape,
                               raw_red_channel, self.__np_red_channel.shape,
                               raw_green_channel, self.__np_green_channel.shape,
                               raw_blue_channel, self.__np_blue_channel.shape,
                               self.__np_threshold, self.__np_filtered,
                               self.__np_filtered.shape))
        # ------------------------------------------------------------------
        # initialize shared image channels ---------------------------------
        self.__shd_red = shared(np.zeros_like(red_indexes,
                                              dtype=config.floatX),
                                borrow=True)

        self.__shd_green = shared(np.zeros_like(green_indexes,
                                                dtype=config.floatX),
                                  borrow=True)

        self.__shd_blue = shared(np.zeros_like(blue_indexes,
                                               dtype=config.floatX),
                                 borrow=True)

        self.__shd_flat_red = shared(np.zeros(
            (1280 * 720), dtype=np.uint32), borrow=True)

        self.__shd_flat_green = shared(np.zeros(
            (1280 * 720), dtype=np.uint32), borrow=True)

        self.__shd_flat_blue = shared(np.zeros(
            (1280 * 720), dtype=np.uint32), borrow=True)
        # ------------------------------------------------------------------

        # initialize filtering shared parameters --------------------------
        self.__shd_params = shared(np.zeros(7, dtype=config.floatX), borrow=True)
        # ------------------------------------------------------------------

        # initialize binary filtering
        # as theano variables ----------------------------------------------

        def image_classification(red, green, blue):
            # shd_params[0] = min_overall
            # shd_params[1] = min_blue
            # shd_params[2] = max_blue
            # shd_params[3] = min_blue_green_diff
            # shd_params[4] = max_blue_green_diff
            # shd_params[5] = min_green_red_diff
            # shd_params[6] = max_green_red_diff
            # shd_params[7] = classification_threshold
            # T.lt(a, b) <------> a < b
            bg_diff = blue - green
            gr_diff = green - red
            return T.cast(
                T.lt(self.__shd_params[7], np.count_nonzero(
                    T.lt(self.__shd_params[1], blue) &
                    T.lt(blue, self.__shd_params[2]) &
                    T.lt(self.__shd_params[0], red) &
                    T.lt(self.__shd_params[0], green) &
                    T.lt(self.__shd_params[3], bg_diff) &
                    T.lt(bg_diff, self.__shd_params[4]) &
                    T.lt(self.__shd_params[5], gr_diff) &
                    T.lt(gr_diff, self.__shd_params[6]))), 'float32')

        def image_filtering(red, green, blue):
            # shd_params[0] = min_overall
            # shd_params[1] = min_blue
            # shd_params[2] = max_blue
            # shd_params[3] = min_blue_green_diff
            # shd_params[4] = max_blue_green_diff
            # shd_params[5] = min_green_red_diff
            # shd_params[6] = max_green_red_diff
            # shd_params[7] = classification_threshold
            # --> [T.lt(a, b)] <--> [a < b] <--
            bg_diff = blue - green
            gr_diff = green - red

            return (
                T.lt(self.__shd_params[1], blue) &
                T.lt(blue, self.__shd_params[2]) &
                T.lt(self.__shd_params[0], red) &
                T.lt(self.__shd_params[0], green) &
                T.lt(self.__shd_params[3], bg_diff) &
                T.lt(bg_diff, self.__shd_params[4]) &
                T.lt(self.__shd_params[5], gr_diff) &
                T.lt(gr_diff, self.__shd_params[6])
            ) * 255.0

        def convert_to_qt_pixel(r, g, b):
            # alpha = 255 -> 255 * (2 ** 24) = 4 278 190 080
            # red -> r * (2 ** 16) = r * 65 536
            # green -> g * (2 ** 8) = g * 256
            return 4278190080 + (b * 65536) + (g * 256) + r


        classification_results, classification_updates = scan(
            image_classification,

            sequences=[self.__shd_red, self.__shd_green, self.__shd_blue]
        )

        filtering_results, filtering_updates = scan(
            image_filtering,
            sequences=[self.__shd_red, self.__shd_green, self.__shd_blue]
        )

        convert_to_qt_results, convert_to_qt_updates = scan(
            convert_to_qt_pixel,
            sequences=[self.__shd_flat_red,
                       self.__shd_flat_green,
                       self.__shd_flat_blue]
        )

        self.__apply_classification_filter = function(
            inputs=[],
            outputs=Out(cuda_ops.gpu_from_host(filtering_results), borrow=True)
        )

        self.__apply_classification = function(
            inputs=[],
            outputs=Out(cuda_ops.gpu_from_host(classification_results), borrow=True)
        )

        self.__convert_to_qt_image = function(
            inputs=[],
            outputs=Out(convert_to_qt_results, borrow=True)
        )
        # -----------------------------------------------------------------

    def __wait_for_multiprocess_cropping(self):
        self.__pool.map_async(func=crop_image_channels,
                              iterable=[0, 1, 2]).get()

    def __wait_for_multiprocess_threshold_classification(self):
        self.__pool.map_async(
            func=classification_by_threshold,
            iterable=self.__threshold_args).get()

    def update_parameters(self, threshold,
                          min_channel_value, min_blue, max_blue,
                          min_blue_green_diff, max_blue_green_diff,
                          min_green_red_diff, max_green_red_diff):

        self.__np_threshold[0] = threshold * 400.0

        self.__shd_params.set_value([
            min_channel_value,
            min_blue,
            max_blue,
            min_blue_green_diff,
            max_blue_green_diff,
            min_green_red_diff,
            max_green_red_diff], borrow=True)

    def apply_filtering(self, image):
        cv2.bilateralFilter(image, 3, 255, 50, dst=self.__np_image)

        self.__wait_for_multiprocess_cropping()

        self.__shd_red.set_value(self.__np_red_channel, borrow=True)
        self.__shd_green.set_value(self.__np_green_channel, borrow=True)
        self.__shd_blue.set_value(self.__np_blue_channel, borrow=True)

        np.copyto(self.__np_filtered,
                  self.__apply_classification_filter())

        self.__wait_for_multiprocess_threshold_classification()

        return self.__np_filtered

    def apply_classification(self, image):
        cv2.bilateralFilter(image, 3, 255, 50, dst=self.__np_image)

        self.__wait_for_multiprocess_cropping()

        self.__shd_red.set_value(self.__np_red_channel, borrow=True)
        self.__shd_green.set_value(self.__np_green_channel, borrow=True)
        self.__shd_blue.set_value(self.__np_blue_channel, borrow=True)

        return self.__apply_classification()

    def crop_to_qt_elements(self, image):
        qt_image = self.convert_to_qt_image(image)

        return np.take(qt_image, )

    def convert_to_qt_image(self, image):
        r, g, b = cv2.split(image)

        self.__shd_flat_red.set_value(r.flatten(), borrow=True)
        self.__shd_flat_green.set_value(g.flatten(), borrow=True)
        self.__shd_flat_blue.set_value(b.flatten(), borrow=True)

        return self.__convert_to_qt_image()


class GpuCameraView(object):
    def __init__(self):
        self.cam = camera.Camera()
        self.cam.start()
        self.frames = self.cam.getCam()
        self.gpu_ops = GpuNumpyQtOperations(1280, 720, 3, 20, 20)

    def raw_view(self):
        pass
        #return self.frames[self.cam.getFrame()][14:, 5:1285]

    def qt_raw_view(self):
        return np.reshape(
            np.uint32(
                self.gpu_ops.convert_to_qt_image(
                    self.frames[self.cam.getFrame()][14:, 5:1285])
            ), (720, 1280))

    def element_view(self):
        pass

    def qt_element_view(self):
        pass

    def __camera_frame(self):
        pass


class QtElement(QWidget):
    def __init__(self, size, parent=None):
        super(QtElement, self).__init__(parent)

        self.w, self.h = size
        self.color = QColor(100, 100, 100)
        self.setFixedSize(self.w + 2, self.h + 2)
        self.image = QImage(self.w, self.h, QImage.Format_RGB32)

        self.__img_arr = np.ndarray(shape=(self.h, self.w), dtype=np.uint32, buffer=self.image.bits())
        np.copyto(self.__img_arr, np.zeros((self.h, self.w), dtype=np.uint32))

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.__draw_element(qp)
        qp.end()

    def __draw_element(self, qp):
        qp.drawImage(1, 1, self.image)
        qp.setPen(self.color)
        qp.drawRect(0, 0, self.w + 1, self.h + 1)

    def numpy_buffer(self):
        return self.__img_arr


class QtWindow(QWidget):
    def __init__(self):
        super(QtWindow, self).__init__()
        self.__cam_view = GpuCameraView()
        self.__element = QtElement((1280, 720))
        self.__init_window()
        self.__display_timer = QTimer(self)
        self.connect(self.__display_timer,
                     SIGNAL("timeout()"),
                     self.display_camera)
        self.__display_timer.start(25)

    def __init_window(self):

        grid = QGridLayout()
        grid.addWidget(self.__element, 0, 0)

        self.setLayout(grid)
        self.setWindowTitle("Camera Mapping Calibration")
        self.show()

    def display_camera(self):
        np.copyto(self.__element.numpy_buffer(),
                  self.__cam_view.qt_raw_view())
        self.__element.repaint()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QtWindow()
    sys.exit(app.exec_())