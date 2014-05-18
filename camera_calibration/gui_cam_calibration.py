from PySide.QtGui import *

import timeit
import camera
import cv2
import os
import sys

import numpy as np


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


def generate_indexes(width, height, crop_width, crop_height):
    indexes = np.arange(width * height).reshape((height, width))
    row_count = height / crop_height
    col_count = width / crop_width

    pixel_indexes = []
    for row in xrange(row_count):
        for col in xrange(col_count):
            pixel_indexes.append(np.asarray(indexes[row * crop_height:(row + 1) * crop_height,
                                 col * crop_width:(col + 1) * crop_width]))

    pixel_indexes = np.asarray(pixel_indexes).reshape((2304, 20, 20, 1))

    return np.concatenate((pixel_indexes * 3, pixel_indexes * 3 + 1, pixel_indexes * 3 + 2), axis=3)


class ElementImage(QWidget):
    def __init__(self, parent=None):
        super(ElementImage, self).__init__(parent)
        self.setFixedSize(22, 22)
        self.image = QImage(20, 20, QImage.Format_RGB32)

        self.img_arr = np.ndarray(shape=(20, 20), dtype=np.uint32, buffer=self.image.bits())
        np.copyto(self.img_arr, np.zeros((20, 20), dtype=np.uint32))

        self.color = QColor(100, 100, 100)

    def enterEvent(self, e):
        self.color = QColor(255, 150, 0)
        self.repaint()

    def leaveEvent(self, e):
        self.color = QColor(100, 100, 100)
        self.repaint()

    def mousePressEvent(self, e):
        self.color = QColor(255, 255, 0)
        self.repaint()

    def mouseReleaseEvent(self, e):
        self.color = QColor(100, 100, 100)
        self.repaint()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.__draw_element(qp)
        qp.end()

    def __draw_element(self, qp):
        qp.drawImage(1, 1, self.image)
        qp.setPen(self.color)
        qp.drawRect(0, 0, 21, 21)

    def numpy_buffer(self):
        return self.img_arr


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.element_images = []
        self.init()

    def init(self):
        def generate_grid_pos(w, h):
            pos = []
            for row in xrange(h):
                for col in xrange(w):
                    pos.append((row, col))
            return pos

        h_count = 36
        w_count = 64
        buffer_indexes = generate_crop_pixels_indexes_3d(1280, 720, 20, 20)

        grid = QGridLayout()
        count = 0
        for coord in generate_grid_pos(w_count, h_count):
            e_img = ElementImage()
            self.element_images.append((e_img, e_img.numpy_buffer(), buffer_indexes[count]))
            grid.addWidget(e_img, coord[0], coord[1])
            count += 1

        grid.setColumnStretch(w_count, 1)
        grid.setSpacing(0)
        grid.setRowStretch(h_count, 1)
        grid.setVerticalSpacing(0)

        self.setLayout(grid)
        self.setGeometry(20, 20, 1550, 850)
        self.setWindowTitle("Camera Mapping Calibration")
        self.show()

        self.display_image()

    def display_image(self):
        img = cv2.cvtColor(cv2.imread(os.path.abspath('test.png')),
                           cv2.COLOR_BGR2RGB)[14:, 5:1285]

        #indexes = generate_indexes(1280, 720, 20, 20)

        row_count = len(img)
        col_count = len(img[0])

        print row_count, col_count

        qt_img = np.zeros((row_count, col_count), dtype=np.uint32)

        #print indexes[0]

        t = timeit.default_timer()
        for row in xrange(row_count):
            for col in xrange(col_count):
                pixel = img[row][col]
                qt_img[row][col] = qRgb(pixel[0],
                                        pixel[1],
                                        pixel[2])



                #for img_element in img[row*20:(row+1)*20, col*20:(col+1)*20]:



            #element_buffer = self.element_images[count][1]


            #for row in xrange(row_count):
            #    for col in xrange(col_count):
            #        pixel = img_element[row][col]
            #        #print pixel, img[row, col]
            #        element_buffer[row][col] = qRgb(pixel[0], pixel[1], pixel[2])
        print 'refreshing elements took...', timeit.default_timer() - t, 'seconds'

        #for row in xrange(row_count):
        #    for col in xrange(col_count):
        #        pixel = img.take(indexes[0])
        #        print pixel.shape


                #element_buffer[row][col] = qRgb(pixel[0], pixel[1], pixel[2])





        """
        t = timeit.default_timer()
        count = 0
        for row in xrange(36):
            for col in xrange(64):
                crop = img[row*20:(row+1)*20, col*20:(col+1)*20]
                pixels = []
                for pixel_row in crop:
                    pixels_col = []
                    for pixel_col in pixel_row:
                        pixels_col.append(qRgb(pixel_col[0], pixel_col[1], pixel_col[2]))
                    pixels.append(pixels_col)
                np.copyto(self.element_images[count][1],  np.asarray(pixels, dtype=np.uint32))
                count += 1
        print 'refreshing elements took...', timeit.default_timer() - t, 'seconds'
        """

        for i in xrange(len(self.element_images)):
            self.element_images[i][0].repaint()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

def testing():
    pass

if __name__ == "__main__":
    testing()
    main()

