
import cPickle
import cv2
import numpy as np
import camera

from PySide.QtGui import *
from PySide.QtCore import *
import PySide.QtCore

import time
import sys
import operator as op
import os

import math

from ngv_filter import *


def custom_property(func):
    f = func()

    def fget(p):
        return p

    def fset(p, value):
        p = value

    def fdel(p):
        del p

    return property(fget, fset, fdel)


def nested_property(func):
    names = func()
    names['doc'] = func.__doc__
    return property(**names)


class GpuViewOperations(object):
    def __init__(self, cam):
        self.cam = cam
        try:
            self.__camera_running = True
            self.cam.start()
            self.frames = self.cam.getCam()
        except RuntimeError:
            self.__camera_running = False
        self.__black_view = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.__image = None

    def __get_next_frame(self):
        if AppState.properties['CalibrationActions']['ShowImageView']:
            return self.__image
        elif self.__camera_running:
            return self.frames[self.cam.getFrame()]
        else:
            return self.__black_view

    def load_image(self, image):
        self.__image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @property
    def camera_object(self):
        return self.cam

    __show_classification_result = False

    @property
    def qt_elements(self):
        def show_weighted_result(np_arr):
            count = 0
            for e in np_arr:
                print count, e
                count += 1

        from ngv_filter import NGVFilter, UtilityOperations
        if AppState.properties['ElementViews']['ShowFilteredElements']:
            if AppState.properties['FilterCalibration'][
                    'ShowWeightedClassification']:
                img = self.__get_next_frame()
                if GpuViewOperations.__show_classification_result:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    show_weighted_result(
                        NGVFilter.elements_weighted_classification(rgb))
                    GpuViewOperations. __show_classification_result = False
                return UtilityOperations.qt_elements_classification(img)
            else:
                return UtilityOperations.qt_elements_filtering(
                    self.__get_next_frame())
        else:
            return UtilityOperations.convert_np_image_to_qt_elements(
                self.__get_next_frame())

    @property
    def qt_image(self):
        if AppState.properties['ElementViews']['ShowFilteredElements']:
            return UtilityOperations.qt_image_filtering(
                self.__get_next_frame())
        else:
            return UtilityOperations.convert_np_image_to_qt_image(
                self.__get_next_frame())

COLORS = {
    'window_bg_color': QColor("#242424"),
    'element_inactive_color': QColor('#6b9fa1'),
    'element_m_over_color': QColor('#9b9d9e'),
    'element_m_pressed_color': QColor('#cc7832'),
    'title_widget_bg_color': QColor('#1f7832'),
    'title_widget_active_bg_color': QColor('#f1f1f1'),
    'title_widget_inactive_bg_color': QColor('#9b9d9e'),
    'title_widget_active_m_over_bg_color': QColor('#fbfbfb'),
    'title_widget_inactive_m_over_bg_color': QColor('#d4d4d4'),
    'calibration_widget_bg_color': QColor('#f1f1f1')
}


class PostfixIncrementable(int):
    def __init__(self, obj):
        super(self.__class__, self)
        self.obj = obj

    def __str__(self):
        return str(self.obj)

    def __get__(self):
        r = self.obj
        print self.obj
        return r

    def __set__(self, value):
        self.obj = value

    def __call__(self, n=1):
        r = self.obj
        self.obj += n
        return r

HEIGHT, WIDTH, DEPTH = 720, 1280, 3
CROP_WIDTH, CROP_HEIGHT = 20, 20
ROW_COUNT = HEIGHT / CROP_HEIGHT
COL_COUNT = WIDTH / CROP_WIDTH

import math

def test_trigo():
    """      #5 1999
            #24 1509
            #26 1117
            #29 798
            #30 525
            #31 405
            #33 483
            #38 841
            #39 344
            #40 691
            #41 933
            #43 2048
    """

    ids = ((5, 1999.0), (24, 1509.0), (26, 1117.0), (29, 798.0), (30, 525.0),
           (31, 405.0), (33, 483.0), (38, 841.0), (39, 344.0), (40, 691.0),
           (41, 933.0), (43, 2048.0))

    rad_90 = op.truediv(math.pi, 2)

    measured = [(AppState.elements[j], j)
                for j in xrange(len(AppState.elements))
                if AppState.elements[j]['Type'] == 1]

    c_h = AppState.properties['MappingCalibration']['CameraHeight']
    c_cam_angle = AppState.properties['MappingCalibration']['CameraAngle']
    print 'Mapping Calibration:\n' \
          'CameraHeight: {} mm | CameraAngle: {} degree\n'.format(
          c_h, c_cam_angle)

    m_h = AppState.properties['RealWorldCalibration']['CameraHeight']
    m_cam_angle = AppState.properties['RealWorldCalibration']['CameraDegreeAngle']
    print 'Real-World Calibration:\n' \
          'CameraHeight: {} mm | CameraAngle: {} degree\n'.format(
          m_h, m_cam_angle)

    print 'Measured Elements:\n'
    for e_, id_ in measured:

        c_sigma = e_['RelativeDegree']
        c_d = e_['RelativeDistance']
        m_d = e_['RealWorldDistance']

        c_id = [_i[0] for _i in ids if _i[1] == m_d]

        c_alpha = rad_90 - math.atan(op.truediv(c_h, c_d))
        m_alpha = rad_90 - math.atan(op.truediv(m_h, m_d))

        estimated = math.tan(c_alpha) * m_h

        print '{}[{}]'.format(c_id, id_)
        print '[Calibration]\tDistance: {} mm | Height: {} mm |' \
              ' Estimated Alpha: {} rad'.format(
              c_d, c_h, c_alpha)
        print '[Measured]\tDistance: {} mm | Height: {} mm |' \
              ' Estimated Alpha: {} rad'.format(
              m_d, m_h, m_alpha)
        print '[Estimated Distance {}] [Error: {}]'.format(
            estimated, 1.0 - (estimated / m_d))
        print '[Calibrated Angle {}] [Error: {}]'.format(
            c_alpha, 1.0 - (c_alpha / m_alpha))

        print ''


class AppState(object):

    application_saved_state_path = os.path.abspath('')
    application_saved_state_file_name = 'AppState.pkl'

    filter_calibration_export_path = os.path.abspath('')
    filter_calibration_file_name = 'filter_calibration.pkl'

    slam_calibration_export_path = os.path.abspath('')
    slam_calibration_file_name = 'slam_calibration.pkl'

    ai_calibration_export_path = os.path.abspath('')
    ai_calibration_file_name = 'ai_calibration.pkl'

    main_window = None
    application = None

    elements = [
        {'NumpyBuffer': None,       # Element Image Buffer
         'QImage': None,            # Qt Image
         'Type': 0,                 # Element Type (see CameraCalibration.Types)
         'RelativeDistance': 0,     # Poster Calibration Distance (mm)
         'RelativeDegree': 0,       # Poster Calibration Angle (degree)
         'RelativeRadianAngle': 0,  # Poster Calibration Angle (radian)
         'RealWorldDistance': 0}    # Real World Calibration Distance (mm)
        for e in xrange((WIDTH * HEIGHT) / (CROP_WIDTH * CROP_HEIGHT))]

    displayed_elements = set()

    selected_element_index = 0

    properties = {
        'FilterCalibration': {
            'ShowWeightedClassification': False,
            'Minimized': True,
            'MinValue': 0,
            'MinBlue': 0,
            'MaxBlue': 0,
            'MinBlueGreenDiff': 0,
            'MaxBlueGreenDiff': 0,
            'MinGreenRedDiff': 0,
            'MaxGreenRedDiff': 0,
            'ThresholdValue': 0
        },
        'CameraCalibration': {
            'Minimized': True,
            'Locked': True,
            'Types': {'Computed Element': 0,    # elements computed from
                                                # measured elements and relative
                                                # value of this element

                      'Measured Element': 1,    # elements measured in real
                                                # world calibration

                      'Range Finder': 2,        # range finder elements computed
                                                # as the origin of the camera
                                                # matrix

                      'Not Processed': 3}       # irrelevant elements that
                                                # should not be processed
        },
        'MappingCalibration': {
            'Minimized': True,
            'Locked': True,
            'CameraAngle': 0,
            'CameraHeight': 0
        },
        'RealWorldCalibration': {
            'Minimized': True,
            'Locked': True,
            'CameraDegreeAngle': 0,
            'CameraRadianAngle': 0,
            'CameraHeight': 0,
            'CameraRangeFinderDistance': 0
        },
        'CameraConfiguration': {
            'Minimized': True
        },
        'ElementViews': {
            'Minimized': True,
            'GridColor': QColor('#6b9fa1'),
            'FreezeCameraView': False,
            'ShowFilteredElements': False,
            'ShowMeasuredElements': False,
            'ShowComputedElements': False,
            'ShowNotProcessedElements': False,
            'ShowRangeFinderElements': False,
            'ShowTypeElementColors': False
        },
        'CalibrationActions': {
            'Minimized': True,
            'ShowImageView': False,
            'LoadedImageViewPath': None
        }
    }

    distance_input_validation = QDoubleValidator(0.0, 10000.0, 4, None)
    distance_input_validation.setNotation(QDoubleValidator.StandardNotation)
    angle_input_validation = QDoubleValidator(0.0, 360.0, 6, None)
    angle_input_validation.setNotation(QDoubleValidator.StandardNotation)

    gpu_view_operations = None

    @staticmethod
    def load_pickle_files():
        def saved_elements_compatible(elements):
            if len(AppState.elements) < 1 and len(elements) < 1:
                return False
            if len(AppState.elements) != len(elements):
                return False

            compatible = True
            for element_key in elements[0].keys():
                if element_key not in AppState.elements[0].keys():
                    compatible = False
                    break
            return compatible

        def saved_properties_compatible(properties):
            compatible = True
            for property_key in properties.keys():
                if not compatible:
                    break
                if property_key not in AppState.properties.keys():
                    compatible = False
                    break

                for p_k in properties[property_key].keys():

                    if p_k not in AppState.properties[property_key].keys():
                        print p_k
                        compatible = False
                        break
            return compatible

        dir_path = AppState.application_saved_state_path
        file_path = dir_path + '/' + AppState.application_saved_state_file_name
        if not os.path.exists(file_path):
            return

        print 'loading state from pickle file... ({})'.format(file_path)
        pkl_file = open(file_path, 'rb')
        state = cPickle.load(pkl_file)
        pkl_file.close()

        assert saved_elements_compatible(state['elements'])
        assert saved_properties_compatible(state['properties'])

        AppState.selected_element_index = state['selected_index']

        for i in xrange(len(state['elements'])):
            e = AppState.elements[i]
            saved_e = state['elements'][i]
            for key, value in saved_e.items():
                e[key] = value

        for k, v in AppState.properties.items():
            for key in v:
                try:
                    if key in AppState.properties[k].keys() \
                            and key in state['properties'][k].keys():
                        AppState.properties[k][key] = state[
                            'properties'][k][key]
                except KeyError:
                    print 'Property [\'{}\'][\'{}\'] not in saved state'.format(
                        key, k)

        f = AppState.properties['CalibrationActions']['LoadedImageViewPath']
        if f is not None and os.path.exists(f):
            AppState.gpu_view_operations.load_image(
                cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))

        AppState.FilterCalibration.update_filter_parameters()
        AppState.ElementViews.update_displayed_elements_set()

        from ngv_filter import NGVFilter, FILTER_CONFIGURATION_FILE
        NGVFilter.load_filter_configurations_from_file(
            FILTER_CONFIGURATION_FILE)

    @staticmethod
    def update_element_real_world_distance(index=None):
        def compute_distance(calibration_height, calibration_distance,
                             measured_height):
            # tan( pi/2 - arctan(hc/dc) ) * mh
            return math.tan(
                op.truediv(math.pi, 2) - math.atan(
                    op.truediv(calibration_height, calibration_distance))) \
                * measured_height \
                if calibration_distance != 0.0 else 0.0

        if index is not None:
            e = AppState.elements[index]
            e['RealWorldDistance'] = compute_distance(
                AppState.properties['MappingCalibration']['CameraHeight'],
                e['RelativeDistance'],
                AppState.properties['RealWorldCalibration']['CameraHeight'])
            if index == AppState.selected_element_index:
                AppState.CameraCalibration.measured_distance_input.setText(
                    str(e['RealWorldDistance']))
        else:
            c_h = AppState.properties['MappingCalibration']['CameraHeight']
            m_h = AppState.properties['RealWorldCalibration']['CameraHeight']
            count = 0
            for e in AppState.elements:
                e['RealWorldDistance'] = compute_distance(
                    c_h, e['RelativeDistance'], m_h)
                if count == AppState.selected_element_index:
                    AppState.CameraCalibration.measured_distance_input.setText(
                        str(e['RealWorldDistance']))
                count += 1

    @staticmethod
    @PySide.QtCore.Slot()
    def save_state_to_pickled_file():
        def remove_unserializable_objects():
            AppState.main_window = None

            for e in AppState.elements:
                e['NumpyBuffer'] = None
                e['QImage'] = None

        remove_unserializable_objects()

        dir_path = AppState.application_saved_state_path
        file_path = dir_path + '/' + AppState.application_saved_state_file_name
        if os.path.exists(file_path):
            os.remove(file_path)

        print 'saving state to pickle file... ({})'.format(file_path)
        state = {
            'elements': AppState.elements,
            'selected_index': AppState.selected_element_index,
            'properties': AppState.properties
        }

        from ngv_filter import NGVFilter, FILTER_CONFIGURATION_FILE
        NGVFilter.save_filter_configurations_to_file(
            FILTER_CONFIGURATION_FILE)

        pkl_file = open(file_path, 'wb')
        cPickle.dump(state, pkl_file, -1)
        pkl_file.close()

    class FilterCalibration(object):

        @staticmethod
        def update_filter_parameters():
            from ngv_filter import NGVFilter, UtilityOperations
            params = AppState.properties['FilterCalibration']
            NGVFilter.update_configurations(
                op.truediv(params['ThresholdValue'], 100),
                params['MinValue'], params['MinBlue'], params['MaxBlue'],
                params['MinBlueGreenDiff'], params['MaxBlueGreenDiff'],
                params['MinGreenRedDiff'], params['MaxGreenRedDiff'])
            UtilityOperations.qt_image_filtering(
                cv2.imread(os.path.abspath('test.png')))

        @staticmethod
        @PySide.QtCore.Slot(int)
        def show_weighted_classification(state):
            if state is 0:
                AppState.properties['FilterCalibration'][
                    'ShowWeightedClassification'] = False
            else:
                AppState.properties['FilterCalibration'][
                    'ShowWeightedClassification'] = True
        show_weighted_classification_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['FilterCalibration']['Minimized'] = \
                not AppState.properties['FilterCalibration']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot(int)
        def min_value_changed(value):
            AppState.properties['FilterCalibration']['MinValue'] = value
            AppState.FilterCalibration.update_filter_parameters()
        minimum_value_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def min_blue_changed(value):
            AppState.properties['FilterCalibration']['MinBlue'] = value
            AppState.FilterCalibration.update_filter_parameters()
        minimum_blue_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def max_blue_changed(value):
            AppState.properties['FilterCalibration']['MaxBlue'] = value
            AppState.FilterCalibration.update_filter_parameters()
        maximum_blue_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def min_blue_green_diff_changed(value):
            AppState.properties['FilterCalibration']['MinBlueGreenDiff'] = value
            AppState.FilterCalibration.update_filter_parameters()
        minimum_blue_green_diff_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def max_blue_green_diff_changed(value):
            AppState.properties['FilterCalibration']['MaxBlueGreenDiff'] = value
            AppState.FilterCalibration.update_filter_parameters()
        maximum_blue_green_diff_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def min_green_red_diff_changed(value):
            AppState.properties['FilterCalibration']['MinGreenRedDiff'] = value
            AppState.FilterCalibration.update_filter_parameters()
        minimum_green_red_diff_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def max_green_red_diff_changed(value):
            AppState.properties['FilterCalibration']['MaxGreenRedDiff'] = value
            AppState.FilterCalibration.update_filter_parameters()
        maximum_green_red_diff_slider = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def threshold_changed(value):
            AppState.properties['FilterCalibration']['ThresholdValue'] = value
            AppState.FilterCalibration.update_filter_parameters()
        classification_threshold_slider = None

    class CameraCalibration(object):

        element_preview = None

        @staticmethod
        def __is_selected_element_measured():
            return AppState.elements[AppState.selected_element_index][
                'Type'] is AppState.properties['CameraCalibration']['Types'][
                    'Measured Element']

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['CameraCalibration']['Minimized'] = \
                not AppState.properties['CameraCalibration']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def lock_unlock():
            AppState.properties['CameraCalibration']['Locked'] = \
                not AppState.properties['CameraCalibration']['Locked']

            if AppState.properties['CameraCalibration']['Locked']:
                AppState.CameraCalibration.lock_button.setText('Unlock')
                AppState.CameraCalibration.distance_input.setReadOnly(False)
                AppState.CameraCalibration.angle_input.setReadOnly(False)
                #AppState.CameraCalibration.measured_distance_input.setReadOnly(
                #    False)
            else:
                AppState.CameraCalibration.lock_button.setText('Lock')
                AppState.CameraCalibration.distance_input.setReadOnly(True)
                AppState.CameraCalibration.angle_input.setReadOnly(True)
                #if AppState.CameraCalibration.__is_selected_element_measured():
                #    AppState.CameraCalibration.measured_distance_input.\
                #        setReadOnly(True)
        lock_button = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def type_selected(index):
            AppState.elements[AppState.selected_element_index]['Type'] = index
            #if AppState.CameraCalibration.__is_selected_element_measured():
            #        AppState.CameraCalibration.measured_distance_input.\
            #            setReadOnly(True)
            AppState.ElementViews.update_displayed_elements_set()
        type_combobox = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def distance_from_rf(text):
            AppState.elements[AppState.selected_element_index][
                'RelativeDistance'] = float(text) if text else 0.0
            AppState.update_element_real_world_distance(
                AppState.selected_element_index)
        distance_input = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def angle_from_rf(text):
            def degree_to_radian(degree_val):
                return op.mul(degree_val, op.truediv(math.pi, 180))

            degree = float(text) if text else 0.0
            AppState.elements[AppState.selected_element_index][
                'RelativeDegree'] = degree

            AppState.elements[AppState.selected_element_index][
                'RelativeRadianAngle'] = degree_to_radian(degree)
        angle_input = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def measured_distance_from_rf(text):
            AppState.elements[AppState.selected_element_index][
                'RealWorldDistance'] = float(text) if text else 0.0
        measured_distance_input = None

        @staticmethod
        def selected_element_changed():
            if AppState.CameraCalibration.__is_selected_element_measured():
                    AppState.CameraCalibration.measured_distance_input.\
                        setReadOnly(True)

        @staticmethod
        def update_element_coordinate():
            l = AppState.CameraCalibration.element_coordinate_label
            index = AppState.selected_element_index
            x = index % COL_COUNT + 1
            y = index / COL_COUNT + 1
            l.setText('<center><font size=4>index: <b>{}</b><br />'
                      'row: {} col: {}</font></center>'.format(
                      AppState.selected_element_index, x, y))
        element_coordinate_label = None

    class MappingCalibration(object):

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['MappingCalibration']['Minimized'] = \
                not AppState.properties['MappingCalibration']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def lock_unlock():
            AppState.properties['MappingCalibration']['Locked'] \
                = not AppState.properties['MappingCalibration']['Locked']
            if AppState.properties['MappingCalibration']['Locked']:
                AppState.MappingCalibration.lock_button.setText('Unlock')
                AppState.MappingCalibration.angle_input.setReadOnly(True)
                AppState.MappingCalibration.height_input.setReadOnly(True)
            else:
                AppState.MappingCalibration.lock_button.setText('Lock')
                AppState.MappingCalibration.angle_input.setReadOnly(False)
                AppState.MappingCalibration.height_input.setReadOnly(False)
        lock_button = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def camera_angle(text):
            AppState.properties['MappingCalibration']['CameraAngle'] \
                = float(text) if text else 0.0
        angle_input = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def camera_height(text):
            AppState.properties['MappingCalibration']['CameraHeight'] \
                = float(text) if text else 0.0
            AppState.update_element_real_world_distance()
        height_input = None

    class RealWorldCalibration(object):

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['RealWorldCalibration']['Minimized'] = \
                not AppState.properties['RealWorldCalibration']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def lock_unlock():
            AppState.properties['RealWorldCalibration']['Locked'] = \
                not AppState.properties['RealWorldCalibration']['Locked']

            if AppState.properties['RealWorldCalibration']['Locked']:
                AppState.RealWorldCalibration.lock_button.setText('Unlock')
                AppState.RealWorldCalibration.angle_input.setReadOnly(True)
                AppState.RealWorldCalibration.distance_input.setReadOnly(True)
                AppState.RealWorldCalibration.height_input.setReadOnly(True)
            else:
                AppState.RealWorldCalibration.lock_button.setText('Lock')
                AppState.RealWorldCalibration.angle_input.setReadOnly(False)
                AppState.RealWorldCalibration.distance_input.setReadOnly(False)
                AppState.RealWorldCalibration.height_input.setReadOnly(False)
        lock_button = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def cam_angle(text):
            def degree_to_radian(degree_val):
                return op.mul(degree_val, op.truediv(math.pi, 180))

            degree = float(text) if text else 0.0
            AppState.properties['RealWorldCalibration']['CameraDegreeAngle'] \
                = degree

            AppState.properties['RealWorldCalibration']['CameraRadianAngle'] \
                = degree_to_radian(degree)
        angle_input = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def camera_height(text):
            AppState.properties['RealWorldCalibration']['CameraHeight'] \
                = float(text) if text else 0.0
            AppState.update_element_real_world_distance()
        height_input = None

        @staticmethod
        @PySide.QtCore.Slot(str)
        def camera_distance_from_rf(text):
            AppState.properties['RealWorldCalibration'][
                'CameraRangeFinderDistance'] = float(text) if text else 0.0
        distance_input = None

    class CameraConfiguration(object):

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['CameraConfiguration']['Minimized'] = \
                not AppState.properties['CameraConfiguration']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def adjust_exposure():
            AppState.gpu_view_operations.camera_object.setExposureMode(
                camera.ExposureMode.AutoOnce)
        exposure_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def adjust_gain():
            AppState.gpu_view_operations.camera_object.setGainMode(
                camera.GainMode.AutoOnce)
        gain_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def adjust_white_balance():
            AppState.gpu_view_operations.camera_object.setWhitebalMode(
                camera.WhitebalMode.AutoOnce)
        white_balance_button = None

    class ElementViews(object):

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['ElementViews']['Minimized'] = \
                not AppState.properties['ElementViews']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def choose_grid_color():
            color = QColorDialog().getColor(
                AppState.properties['ElementViews']['GridColor'],
                AppState.main_window, 'Choose Grid Color')
            if color.isValid():
                AppState.properties['ElementViews']['GridColor'] = color
                AppState.main_window.repaint()
            AppState.ElementViews.update_displayed_elements_set()
        grid_color_picker_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def freeze_camera_view(state):
            if state is 0:
                AppState.properties['ElementViews']['FreezeCameraView'] = False
            else:
                AppState.properties['ElementViews']['FreezeCameraView'] = True
            AppState.ElementViews.update_displayed_elements_set()
        freeze_cam_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot(int)
        def show_filtered_elements(state):
            if state is 0:
                AppState.properties['ElementViews'][
                    'ShowFilteredElements'] = False
            else:
                AppState.properties['ElementViews'][
                    'ShowFilteredElements'] = True
            AppState.ElementViews.update_displayed_elements_set()
        filtered_elements_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot()
        def show_measured_elements(state):
            if state is 0:
                AppState.properties['ElementViews'][
                    'ShowMeasuredElements'] = False
            else:
                AppState.properties['ElementViews'][
                    'ShowMeasuredElements'] = True
            AppState.ElementViews.update_displayed_elements_set()
        show_measured_elements_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot()
        def show_computed_elements(state):
            if state is 0:
                AppState.properties['ElementViews'][
                    'ShowComputedElements'] = False
            else:
                AppState.properties['ElementViews'][
                    'ShowComputedElements'] = True
            AppState.ElementViews.update_displayed_elements_set()
        show_computed_elements_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot()
        def show_not_processed_elements(state):
            if state is 0:
                AppState.properties['ElementViews'][
                    'ShowNotProcessedElements'] = False
            else:
                AppState.properties['ElementViews'][
                    'ShowNotProcessedElements'] = True
            AppState.ElementViews.update_displayed_elements_set()
        show_not_processed_elements_checkbox = None

        @staticmethod
        @PySide.QtCore.Slot()
        def show_range_finder_elements(state):
            if state is 0:
                AppState.properties['ElementViews'][
                    'ShowRangeFinderElements'] = False
            else:
                AppState.properties['ElementViews'][
                    'ShowRangeFinderElements'] = True
            AppState.ElementViews.update_displayed_elements_set()
        show_range_finder_elements_checkbox = None

        @staticmethod
        def update_displayed_elements_set():
            types = AppState.properties['CameraCalibration']['Types']
            l = [None for t in types]
            for t, i in types.items():
                if t == 'Computed Element':
                    l[i] = AppState.properties['ElementViews'][
                        'ShowComputedElements']
                elif t == 'Measured Element':
                    l[i] = AppState.properties['ElementViews'][
                        'ShowMeasuredElements']
                elif t == 'Range Finder':
                    l[i] = AppState.properties['ElementViews'][
                        'ShowRangeFinderElements']
                elif t == 'Not Processed':
                    l[i] = AppState.properties['ElementViews'][
                        'ShowNotProcessedElements']
            count = 0
            AppState.displayed_elements.clear()
            for e in AppState.elements:
                if l[e['Type']]:
                    AppState.displayed_elements.add(count)
                count += 1

    class CalibrationActions(object):

        @staticmethod
        @PySide.QtCore.Slot()
        def minimized():
            AppState.properties['CalibrationActions']['Minimized'] = \
                not AppState.properties['CalibrationActions']['Minimized']

        @staticmethod
        @PySide.QtCore.Slot()
        def new_calibration():
            print 'new calibration'
        new_calibration_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def save_calibration():
            print 'save calibration'
        save_calibration_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def load_calibration():
            print 'load calibration'
        load_calibration_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def save_view_as_image():
            print 'save view as image'
        save_view_as_image_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def load_view_from_camera():
            AppState.properties['CalibrationActions']['ShowImageView'] = False
        load_view_from_camera_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def load_view_from_image():
            f = str(QFileDialog.getOpenFileName()[0])
            if f:
                AppState.gpu_view_operations.load_image(cv2.cvtColor(
                    cv2.imread(f), cv2.COLOR_BGR2RGB))
                AppState.properties['CalibrationActions'][
                    'LoadedImageViewPath'] = f
                AppState.properties['CalibrationActions'][
                    'ShowImageView'] = True
        load_view_from_image_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def export_filter_calibration():
            print 'export filter calibration'
        export_filter_calibration_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def export_slam_calibration():
            f = str(QFileDialog.getSaveFileName()[0])
            if f:
                from ngv_filter import NGVFilter
                NGVFilter.save_elements_structure(f, AppState.elements)
        export_slam_calibration_button = None

        @staticmethod
        @PySide.QtCore.Slot()
        def export_ai_calibration():
            print 'export AI calibration'
        export_ai_calibration_button = None


class QtWindow(QWidget):
    class CameraWidget(QWidget):
        class QtElement(QWidget):
            def __init__(self, element_id, selected_function,
                         size, parent=None):
                super(QtWindow.CameraWidget.QtElement, self).__init__(parent)
                self.__w, self.__h = size
                self.setFixedSize(self.__w + 2, self.__h + 2)

                self.__inactive_color = COLORS['element_inactive_color']
                self.__m_over_color = COLORS['element_m_over_color']
                self.__m_pressed_color = COLORS['element_m_pressed_color']
                self.__color = self.__inactive_color

                element = AppState.elements[element_id]

                self.__image = element['QImage'] = QImage(self.__w, self.__h,
                                                          QImage.Format_RGB32)
                self.__img_arr = element['NumpyBuffer'] = \
                    np.ndarray(shape=(self.__h, self.__w),
                               dtype=np.uint32,
                               buffer=self.__image.bits())

                np.copyto(self.__img_arr, np.zeros((self.__h, self.__w),
                                                   dtype=np.uint32))

                self.__selected_f = selected_function
                self.__element_id = element_id
                self.display_element = False

            def enterEvent(self, e):
                self.__color = self.__m_over_color
                self.repaint()

            def leaveEvent(self, e):
                self.__color = self.__inactive_color
                self.repaint()

            def mousePressEvent(self, e):
                ##mouse left event Qpoint Point mouse right  event
                #print e.button()
                self.__color = self.__m_pressed_color
                self.repaint()

            def mouseReleaseEvent(self, e):
                self.__color = self.__inactive_color
                self.repaint()
                self.__selected_f(self.__element_id)

            def paintEvent(self, e):
                qp = QPainter()
                qp.begin(self)
                if self.display_element:
                    self.__draw_element(qp)
                qp.end()

            def __draw_element(self, qp):
                qp.drawImage(1, 1, self.__image)
                if self.__color is self.__inactive_color \
                        and self.__inactive_color is \
                        not AppState.properties['ElementViews']['GridColor']:
                    self.__color = self.__inactive_color = \
                        AppState.properties['ElementViews']['GridColor']

                if self.__element_id is not AppState.selected_element_index:
                    qp.setPen(self.__color)
                else:
                    qp.setPen(self.__color.lighter(220))

                qp.drawRect(0, 0, self.__w + 1, self.__h + 1)

            def numpy_buffer(self):
                return self.__img_arr

        def __init__(self, selected_element_function):
            super(QtWindow.CameraWidget, self).__init__()
            self.__elements = []
            self.__selected_element_f = selected_element_function

            grid = QGridLayout()

            self.__add_elements_to_grid(grid, 36, 64)

            grid.setColumnStretch(36, 1)
            grid.setRowStretch(64, 1)
            grid.setSpacing(0)
            grid.setVerticalSpacing(0)
            grid.setHorizontalSpacing(0)
            grid.setContentsMargins(0, 1, 0, 0)

            self.setLayout(grid)

        def __add_elements_to_grid(self, grid, h_count, w_count):
            def generate_grid_pos(w, h):
                pos = []
                for row in xrange(h):
                    for col in xrange(w):
                        pos.append((row, col))
                return pos

            count = 0
            for coord in generate_grid_pos(w_count, h_count):
                e_img = QtWindow.CameraWidget.QtElement(
                    count, self.__selected_element_f, (20, 20))
                self.__elements.append((e_img, e_img.numpy_buffer()))
                grid.addWidget(e_img, coord[0], coord[1])
                count += 1

        def refresh_camera_widget(self):
            element_views = AppState.gpu_view_operations.qt_elements
            #t = timeit.default_timer()
            for i in xrange(len(self.__elements)):
                element = self.__elements[i]
                element[0].display_element = i in AppState.displayed_elements
                if element[0].display_element:
                    np.copyto(element[1], element_views[i])
            self.repaint()
            #print 'copying elements took...', timeit.default_timer() - t

    class UserInputWidget(QWidget):
        class InputWithUnit(QWidget):
            def __init__(self, q_line_edit, unit_text):
                super(QtWindow.UserInputWidget.InputWithUnit, self).__init__()
                self.__line_edit = q_line_edit
                self.__unit_label = QLabel(unit_text)
                self.__unit_label.setMargin(2)
                grid = QGridLayout()
                grid.addWidget(self.__line_edit, 0, 0)
                grid.addWidget(self.__unit_label, 0, 1)
                grid.setColumnStretch(2, 1)
                grid.setVerticalSpacing(0)
                grid.setHorizontalSpacing(0)
                grid.setSpacing(0)
                grid.setContentsMargins(0, 0, 0, 0)
                self.setLayout(grid)

        class CalibrationWidget(QWidget):
            class TitleWidget(QPushButton):
                def __init__(self, title, child_widget, pressed_slot,
                             active, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          TitleWidget, self).__init__("", parent)
                    self.setText(title)
                    self.setFlat(True)

                    self.__active = active
                    self.setFont(QFont('Cantarell', 11))

                    self.__active_palette = QPalette()
                    self.__active_palette.setColor(
                        self.backgroundRole(),
                        COLORS['title_widget_active_bg_color']
                    )

                    self.__inactive_palette = QPalette()
                    self.__inactive_palette.setColor(
                        self.backgroundRole(),
                        COLORS['title_widget_inactive_bg_color']
                    )

                    self.__active_m_over_palette = QPalette()
                    self.__active_m_over_palette.setColor(
                        self.backgroundRole(),
                        COLORS['title_widget_active_m_over_bg_color']
                    )

                    self.__inactive_m_over_palette = QPalette()
                    self.__inactive_m_over_palette.setColor(
                        self.backgroundRole(),
                        COLORS['title_widget_inactive_m_over_bg_color']
                    )

                    self.__child = child_widget
                    self.__pressed = pressed_slot

                    if self.__active:
                        self.setPalette(self.__active_palette)
                    else:
                        self.setPalette(self.__inactive_palette)
                    self.setAutoFillBackground(True)

                def enterEvent(self, e):
                    if self.__active:
                        self.setPalette(self.__active_m_over_palette)
                    else:
                        self.setPalette(self.__inactive_m_over_palette)

                    self.repaint()

                def leaveEvent(self, e):
                    if self.__active:
                        self.setPalette(self.__active_palette)
                    else:
                        self.setPalette(self.__inactive_palette)
                    self.repaint()

                def mousePressEvent(self, e):
                    self.__active = not self.__active
                    if self.__active:
                        self.setPalette(self.__active_palette)
                    else:
                        self.setPalette(self.__inactive_palette)
                    self.__pressed()
                    self.__child.setVisible(self.__active)
                    self.repaint()

            class FilterCalibration(QWidget):
                class CustomSlider(QSlider):
                    def __init__(self, initial_position, min_max_range):
                        super(self.__class__, self).__init__()

                        self.__initial_pos = initial_position
                        self.__pixel_ratio = None
                        self.__pixel_translation = None
                        self.__last_size = self.size()

                        self.__value_lbl = QLabel('', self)
                        self.__value_lbl.setMinimumWidth(30)
                        self.__last_lbl_size = self.__value_lbl.size()

                        self.__first_pass = True

                        self.setFocusPolicy(Qt.NoFocus)
                        self.setRange(min_max_range[0], min_max_range[1])
                        self.setSingleStep(0)
                        self.setSliderPosition(self.__initial_pos)
                        self.setPageStep(1)
                        self.setFixedSize(82, 255)

                    def paintEvent(self, e):
                        super(self.__class__, self).paintEvent(e)
                        if self.__first_pass:
                            self.setSliderPosition(self.__initial_pos)
                            self.sliderChange(QAbstractSlider.SliderValueChange)
                            self.__first_pass = False

                    def sliderChange(self, e):
                        super(self.__class__, self).sliderChange(e)
                        if self.__pixel_ratio is None \
                                or self.__last_size is not self.size() \
                                or self.__last_lbl_size is \
                                not self.__value_lbl.size():
                            self.__last_size = self.size()
                            self.__last_lbl_size = self.__value_lbl.size()
                            self.__pixel_ratio = (
                                -self.size().height() +
                                self.__value_lbl.size().height() + 4)

                            self.__pixel_translation = (
                                self.size().height() -
                                self.__value_lbl.size().height() - 2)

                        ratio = op.truediv(self.value() - self.minimum(),
                                           self.maximum() - self.minimum())

                        self.__value_lbl.move(2,
                                              self.__pixel_ratio * ratio +
                                              self.__pixel_translation)
                        self.__value_lbl.setText(unicode(self.value()))

                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          FilterCalibration, self).__init__()

                    def create_frame(l, s):
                        _frame = QFrame()

                        _grid = QGridLayout()
                        _grid.addWidget(l, 0, 0)
                        _grid.addWidget(s, 1, 0)
                        _grid.setSpacing(0)
                        _grid.setHorizontalSpacing(0)
                        _grid.setVerticalSpacing(0)
                        _grid.setContentsMargins(1, 5, 0, 5)

                        _frame.setLayout(_grid)
                        _frame.setLineWidth(1)
                        _frame.setMidLineWidth(1)
                        _frame.setFrameShadow(QFrame.Raised)
                        _frame.setFrameShape(QFrame.StyledPanel)

                        _p = QPalette()
                        _p.setColor(self.backgroundRole(), QColor('#f0f0f0'))
                        _frame.setPalette(_p)
                        _frame.setAutoFillBackground(True)
                        return _frame

                    grid = QGridLayout()
                    #
                    # QCheckBox + QLabel -> Show Weighted Classification
                    cb = AppState.FilterCalibration.\
                        show_weighted_classification_checkbox = QCheckBox()
                    cb.stateChanged.connect(AppState.FilterCalibration.
                                            show_weighted_classification)
                    if AppState.properties['FilterCalibration'][
                            'ShowWeightedClassification']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    grid.addWidget(cb, 0, 0)

                    lbl = QLabel('<font size=2>Show<br />'
                                 'Binary<br />Classification</font>')
                    grid.addWidget(lbl, 0, 0, alignment=3)
                    #
                    # QSlider -> Classification Threshold Value
                    label = QLabel('<center><font size=2>'
                                   'Classification<br>Threshold'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'ThresholdValue']

                    slider = AppState.FilterCalibration.\
                        classification_threshold_slider = \
                        self.CustomSlider(initial_value, (0, 100))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.threshold_changed)

                    grid.addWidget(create_frame(label, slider), 1, 0)
                    #
                    # QLabel + QSlider -> Minimum Channel Value
                    label = QLabel('<center><font size=2>'
                                   'Minimum<br>Color Value'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MinValue']

                    slider = AppState.FilterCalibration.minimum_value_slider \
                        = self.CustomSlider(initial_value, (0, 255))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.min_value_changed)

                    grid.addWidget(create_frame(label, slider), 1, 1)
                    #
                    # QLabel + QSlider -> Minimum Blue Value
                    label = QLabel('<center><font size=2>'
                                   'Minimum<br>Blue Value'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MinBlue']

                    slider = AppState.FilterCalibration.minimum_blue_slider \
                        = self.CustomSlider(initial_value, (0, 255))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.min_blue_changed)

                    grid.addWidget(create_frame(label, slider), 2, 0)
                    #
                    # QLabel + QSlider -> Maximum Blue Value
                    label = QLabel('<center><font size=2>'
                                   'Maximum<br>Blue Value'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MaxBlue']

                    slider = AppState.FilterCalibration.maximum_blue_slider \
                        = self.CustomSlider(initial_value, (0, 255))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.max_blue_changed)

                    grid.addWidget(create_frame(label, slider), 2, 1)
                    #
                    # QLabel + QSlider -> Minimum Blue Green Difference Value
                    label = QLabel('<center><font size=2>'
                                   'Minimum<br>Blue Green<br>Difference'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MinBlueGreenDiff']

                    slider = AppState.FilterCalibration.\
                        minimum_blue_green_diff_slider = \
                        self.CustomSlider(initial_value, (-100, 150))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.min_blue_green_diff_changed)

                    grid.addWidget(create_frame(label, slider), 3, 0)
                    #
                    # QLabel + QSlider -> Maximum Blue Green Difference Value
                    label = QLabel('<center><font size=2>'
                                   'Maximum<br>Blue Green<br>Difference'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MaxBlueGreenDiff']

                    slider = AppState.FilterCalibration.\
                        maximum_blue_green_diff_slider = \
                        self.CustomSlider(initial_value, (-100, 150))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.max_blue_green_diff_changed)

                    grid.addWidget(create_frame(label, slider), 3, 1)
                    #
                    # QLabel + QSlider -> Minimum Green Red Difference Value
                    label = QLabel('<center><font size=2>'
                                   'Minimum<br>Green Red<br>Difference'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MinGreenRedDiff']

                    slider = AppState.FilterCalibration.\
                        minimum_green_red_diff_slider = \
                        self.CustomSlider(initial_value, (-100, 150))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.min_green_red_diff_changed)

                    grid.addWidget(create_frame(label, slider), 4, 0)
                    #
                    # QLabel + QSlider -> Minimum Green Red Difference Value
                    label = QLabel('<center><font size=2>'
                                   'Maximum<br>Green Red<br>Difference'
                                   '</font></center>')

                    initial_value = AppState.properties['FilterCalibration'][
                        'MaxGreenRedDiff']

                    slider = AppState.FilterCalibration.\
                        maximum_green_red_diff_slider = \
                        self.CustomSlider(initial_value, (-100, 150))

                    slider.valueChanged[int].connect(
                        AppState.FilterCalibration.max_green_red_diff_changed)

                    grid.addWidget(create_frame(label, slider), 4, 1)

                    grid.setHorizontalSpacing(1)
                    grid.setVerticalSpacing(1)
                    grid.setContentsMargins(1, 2, 1, 2)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['FilterCalibration']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            class CamCalibration(QWidget):
                class ZoomedElement(QWidget):
                    def __init__(self):
                        super(QtWindow.UserInputWidget.CalibrationWidget.
                              CamCalibration.ZoomedElement, self).__init__()
                        self.__w, self.__h = 60, 60
                        self.setFixedSize(self.__w + 2, self.__h + 2)

                        self.__color = COLORS['element_inactive_color']

                        self.__image = QImage(CROP_WIDTH, CROP_HEIGHT,
                                              QImage.Format_RGB32)

                    def q_image(self, image=None):
                        if image is not None:
                            self.__image = image
                        return self.__image

                    def paintEvent(self, e):
                        qp = QPainter()
                        qp.begin(self)
                        qp.drawImage(1, 1, self.__image.scaled(
                            self.__w, self.__h, mode=Qt.SmoothTransformation))
                        qp.setPen(self.__color)
                        qp.drawRect(0, 0, self.__w + 1, self.__h + 1)
                        qp.end()

                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          CamCalibration, self).__init__()

                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    grid.setColumnStretch(0, 1)
                    #
                    # QPushButton -> Camera Mapping Calibration
                    #                Lock/Unlock with confirmation
                    #b = AppState.CameraCalibration.lock_button = \
                    #    QPushButton('Unlock' if AppState.properties[
                    #        'CameraCalibration']['Locked'] else 'Lock')
                    #b.pressed.connect(AppState.CameraCalibration.lock_unlock)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QComboBox Types -> Computed Element, RangeFinder,
                    #                    Not Processed, Measured Element
                    cb = AppState.CameraCalibration.type_combobox = QComboBox()

                    types = AppState.properties['CameraCalibration']['Types']
                    l = [None for t in types]
                    for t, i in types.items():
                        l[i] = t
                    cb.addItems(l)

                    cb.currentIndexChanged.connect(
                        AppState.CameraCalibration.type_selected)

                    cb.wheelEvent = lambda e: e.ignore()

                    grid.addWidget(cb, row(+1), 0)
                    #
                    # QWidget -> 3x Zoomed Preview
                    w = AppState.CameraCalibration.element_preview = \
                        self.ZoomedElement()

                    grid.addWidget(w, row(+1), 0, alignment=4)
                    #
                    # QLabel -> Element X, Y Position in the Grid
                    l = AppState.CameraCalibration.element_coordinate_label = \
                        QLabel('')

                    grid.addWidget(l, row(+1), 0, alignment=4)
                    # QLabel + QLineEdit -> Distance
                    le = AppState.CameraCalibration.distance_input = QLineEdit()
                    le.textChanged.connect(
                        AppState.CameraCalibration.distance_from_rf)

                    le.setValidator(AppState.distance_input_validation)
                    le.setMaximumWidth(100)

                    self.__distance_label = QLabel(
                        '<font size=3><b>Distance</b></font>'
                        '<br><font size=2>from center of Range Finder:</font>')

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=3><b>mm</b><font>')

                    grid.addWidget(self.__distance_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Angle (degree)
                    le = AppState.CameraCalibration.angle_input = QLineEdit()
                    le.textChanged.connect(
                        AppState.CameraCalibration.angle_from_rf)

                    le.setValidator(AppState.angle_input_validation)
                    le.setMaximumWidth(70)

                    self.__angle_degree_label = QLabel(
                        '<font size=3><b>Angle</b></font>'
                        '<br><font size=2>from center of Range Finder:</font>')

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=5>&deg;<font>')

                    grid.addWidget(self.__angle_degree_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Measured distance
                    #                       from range finder
                    # (Locked is element is a not a measured element)
                    le = AppState.CameraCalibration.measured_distance_input = \
                        QLineEdit()
                    le.textChanged.connect(
                        AppState.CameraCalibration.measured_distance_from_rf)
                    le.setReadOnly(AppState.elements[
                        AppState.selected_element_index]['Type'] != 1)
                    le.setValidator(AppState.distance_input_validation)
                    le.setMaximumWidth(120)

                    self.__real_world_distance_label = QLabel(
                        '<font size=3><b>Real-World Distance</b></font>'
                        '<br><font size=2>from center of Range Finder:</font>')

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=3><b>mm</b><font>')

                    grid.addWidget(self.__real_world_distance_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['CameraCalibration']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

                    self.refresh_calibration()

                def refresh_calibration(self):
                    element_id = AppState.selected_element_index
                    selected_element = AppState.elements[element_id]

                    AppState.CameraCalibration.element_preview.q_image(
                        selected_element['QImage'])

                    AppState.CameraCalibration.update_element_coordinate()

                    AppState.CameraCalibration.type_combobox.setCurrentIndex(
                        selected_element['Type'])

                    AppState.CameraCalibration.distance_input.setText(
                        str(selected_element['RelativeDistance']))

                    AppState.CameraCalibration.angle_input.setText(
                        str(selected_element['RelativeDegree']))

                    AppState.CameraCalibration.measured_distance_input.setText(
                        str(selected_element['RealWorldDistance']))

                    AppState.CameraCalibration.selected_element_changed()

                    self.repaint()

            class MappingCalibration(QWidget):
                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          MappingCalibration, self).__init__()
                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    grid.setColumnStretch(0, 1)
                    #
                    # QPushButton -> Real World Mapping calibration
                    #                Lock/Unlock with confirmation
                    #b = AppState.MappingCalibration.lock_button \
                    #    = QPushButton('Unlock' if AppState.properties[
                    #        'MappingCalibration']['Locked'] else 'Lock')
                    #b.pressed.connect(AppState.MappingCalibration.lock_unlock)
                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Camera Angle (degree)
                    le = AppState.MappingCalibration.angle_input = QLineEdit()
                    le.textChanged.connect(
                        AppState.MappingCalibration.camera_angle)
                    le.setValidator(AppState.angle_input_validation)
                    le.setMaximumWidth(70)
                    le.setText(str(AppState.properties['MappingCalibration'][
                        'CameraAngle']))

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=5>&deg;<font>')

                    self.__angle_label = QLabel(
                        '<font size=3><b>Camera Angle</b></font>'
                        '<br><font size=2>from Poster:</font>')

                    grid.addWidget(self.__angle_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Height from ground
                    le = AppState.MappingCalibration.height_input = QLineEdit()
                    le.textChanged.connect(AppState.MappingCalibration.
                                           camera_height)
                    le.setValidator(AppState.distance_input_validation)
                    le.setMaximumWidth(100)
                    le.setText(str(AppState.properties['MappingCalibration'][
                        'CameraHeight']))

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=3><b>mm</b><font>')

                    self.__height_label = QLabel(
                        '<font size=3><b>Camera Height</b></font>'
                        '<br><font size=2>from Poster:</font>')

                    grid.addWidget(self.__height_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['MappingCalibration']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            class RealWorldCalibration(QWidget):
                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          RealWorldCalibration, self).__init__()

                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    grid.setColumnStretch(0, 1)
                    #
                    # QPushButton -> Real World Mapping calibration
                    #                Lock/Unlock with confirmation
                    #b = AppState.RealWorldCalibration.lock_button = \
                    #    QPushButton('Unlock' if AppState.properties[
                    #        'RealWorldCalibration']['Locked'] else 'Lock')
                    #b.pressed.connect(AppState.RealWorldCalibration.lock_unlock)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Camera Angle (degree)
                    le = AppState.RealWorldCalibration.angle_input = QLineEdit()
                    le.textChanged.connect(AppState.RealWorldCalibration.
                                           cam_angle)
                    le.setValidator(AppState.angle_input_validation)
                    le.setMaximumWidth(70)
                    le.setText(str(AppState.properties['RealWorldCalibration'][
                        'CameraDegreeAngle']))

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=5>&deg;<font>')

                    self.__angle_degree_label = QLabel(
                        '<font size=3><b>Camera Angle</b></font>'
                        '<br><font size=2>from Ground:</font>')

                    grid.addWidget(self.__angle_degree_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)
                    #
                    # QLabel + QInputBox -> Height from ground
                    le = AppState.RealWorldCalibration.height_input = \
                        QLineEdit()
                    le.textChanged.connect(AppState.RealWorldCalibration.
                                           camera_height)
                    le.setValidator(AppState.distance_input_validation)
                    le.setMaximumWidth(100)
                    le.setText(str(AppState.properties['RealWorldCalibration'][
                        'CameraHeight']))

                    iwu = QtWindow.UserInputWidget.InputWithUnit(
                        le, '<font size=3><b>mm</b><font>')

                    self.__camera_height_label = QLabel(
                        '<font size=3><b>Camera Height</b></font>'
                        '<br><font size=2>from Ground:</font>')

                    grid.addWidget(self.__camera_height_label, row(+1), 0)
                    grid.addWidget(iwu, row(+1), 0)
                    #
                    # QLabel + QInputBox -> distance from Range Finder
                    #le = AppState.RealWorldCalibration.distance_input =\
                    #    QLineEdit()
                    #le.textChanged.connect(AppState.RealWorldCalibration.
                    #                       camera_distance_from_rf)
                    #le.setValidator(AppState.distance_input_validation)
                    #le.setMaximumWidth(100)
                    #le.setText(str(AppState.properties['RealWorldCalibration'][
                    #    'CameraRangeFinderDistance']))

                    #iwu = QtWindow.UserInputWidget.InputWithUnit(
                    #    le, '<font size=3><b>mm</b><font>')

                    #self.__cam_rf_distance_label = QLabel(
                    #    '<font size=3><b>Camera Ground Distance</b></font>'
                    #    '<br><font size=2>from Range Finder:</font>')

                    #grid.addWidget(self.__cam_rf_distance_label, row(+1), 0)
                    #grid.addWidget(iwu, row(+1), 0)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['RealWorldCalibration']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            class CameraConfiguration(QWidget):
                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          CameraConfiguration, self).__init__()

                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    grid.setColumnStretch(0, 1)
                    #
                    # QPushButton -> AutoOnce Exposure
                    b = AppState.CameraConfiguration.exposure_button = \
                        QPushButton("Adjust Exposure")
                    b.pressed.connect(AppState.CameraConfiguration.
                                      adjust_exposure)

                    grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> AutoOnce Gain
                    b = AppState.CameraConfiguration.gain_button = QPushButton(
                        "Adjust Gain")
                    b.pressed.connect(AppState.CameraConfiguration.adjust_gain)

                    grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> AutoOnce WhiteBalance
                    b = AppState.CameraConfiguration.white_balance_button = \
                        QPushButton("Adjust White Balance")
                    b.pressed.connect(AppState.CameraConfiguration.
                                      adjust_white_balance)

                    grid.addWidget(b, row(+1), 0)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['CameraConfiguration']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            class ElementViews(QWidget):
                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          ElementViews, self).__init__()

                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    #
                    # QCheckBox -> Freeze Camera View
                    #cb = AppState.ElementViews.freeze_cam_checkbox = QCheckBox(
                    #    'Freeze Camera View')
                    #cb.stateChanged.connect(AppState.ElementViews.
                    #                        freeze_camera_view)
                    #if AppState.properties['ElementViews']['FreezeCameraView']:
                    #    cb.setCheckState(Qt.Checked)
                    #else:
                    #    cb.setCheckState(Qt.Unchecked)
                    #grid.addWidget(cb, row(+1), 0)
                    #
                    # JCheckedButton -> View Filtered Elements
                    cb = AppState.ElementViews.filtered_elements_checkbox = \
                        QCheckBox('Show Filtered\nElements')
                    cb.stateChanged.connect(AppState.ElementViews.
                                            show_filtered_elements)
                    if AppState.properties['ElementViews'][
                            'ShowFilteredElements']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    grid.addWidget(cb, row(+1), 0)
                    #
                    # JCheckedButton -> View Measured Elements
                    cb = AppState.ElementViews.\
                        show_measured_elements_checkbox = QCheckBox(
                            'Show Measured\nElements')
                    cb.stateChanged.connect(AppState.ElementViews.
                                            show_measured_elements)
                    if AppState.properties['ElementViews'][
                            'ShowMeasuredElements']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    grid.addWidget(cb, row(+1), 0)
                    #
                    # QCheckedButton -> View Computed Elements
                    cb = AppState.ElementViews.\
                        show_computed_elements_checkbox = QCheckBox(
                            'Show Computed\nElements')
                    if AppState.properties['ElementViews'][
                            'ShowComputedElements']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    cb.stateChanged.connect(AppState.ElementViews.
                                            show_computed_elements)

                    grid.addWidget(cb, row(+1), 0)
                    #
                    # QCheckedButton -> View Not Processed Elements
                    cb = AppState.ElementViews.\
                        show_not_processed_elements_checkbox = QCheckBox(
                            'Show Not Processed\nElements')
                    cb.stateChanged.connect(AppState.ElementViews.
                                            show_not_processed_elements)
                    if AppState.properties['ElementViews'][
                            'ShowNotProcessedElements']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    grid.addWidget(cb, row(+1), 0)
                    #
                    # QCheckedButton -> View Range Finder Elements
                    cb = AppState.ElementViews.\
                        show_range_finder_elements_checkbox = QCheckBox(
                            'Show Range Finder\nElements')
                    cb.stateChanged.connect(AppState.ElementViews.
                                            show_range_finder_elements)
                    if AppState.properties['ElementViews'][
                            'ShowRangeFinderElements']:
                        cb.setCheckState(Qt.Checked)
                    else:
                        cb.setCheckState(Qt.Unchecked)
                    grid.addWidget(cb, row(+1), 0)
                    #
                    # QPushButton -> Choose Color
                    b = AppState.ElementViews.grid_color_picker_button = \
                        QPushButton("Choose Grid Color")

                    b.pressed.connect(AppState.ElementViews.choose_grid_color)

                    grid.addWidget(b, row(+1), 0)

                    grid.setHorizontalSpacing(0)
                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['ElementViews']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            class CalibrationActions(QWidget):
                def __init__(self, parent=None):
                    super(QtWindow.UserInputWidget.CalibrationWidget.
                          CalibrationActions, self).__init__()

                    spacer = QSpacerItem(2, 15)

                    grid = QGridLayout()
                    row = PostfixIncrementable(0)
                    #
                    # QPushButton -> New Calibration
                    #b = AppState.CalibrationActions.new_calibration_button = \
                    #    QPushButton('New Calibration')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  new_calibration)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Save Calibration
                    #b = AppState.CalibrationActions.save_calibration_button = \
                    #    QPushButton('Save Calibration')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  save_calibration)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Load Calibration
                    #b = AppState.CalibrationActions.load_calibration_button = \
                    #    QPushButton('Load Calibration')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  load_calibration)

                    #grid.addWidget(b, row(+1), 0)
                    #grid.addItem(spacer, row(+1), 0)
                    #
                    # QPushButton -> Save View As Image
                    #b = AppState.CalibrationActions.save_view_as_image_button =\
                    #    QPushButton('Save View As Image')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  save_view_as_image)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Load View From Camera
                    b = AppState.CalibrationActions.\
                        load_view_from_camera_button = QPushButton(
                            'Load View From Camera')
                    b.pressed.connect(AppState.CalibrationActions.
                                      load_view_from_camera)

                    grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Load View From Image
                    b = AppState.CalibrationActions.\
                        load_view_from_image_button = QPushButton(
                            'Load View From Image')
                    b.pressed.connect(AppState.CalibrationActions.
                                      load_view_from_image)

                    grid.addWidget(b, row(+1), 0)
                    grid.addItem(spacer, row(+1), 0)
                    #
                    # QPushButton -> Export Serialization For Binary Filter
                    #b = AppState.CalibrationActions.\
                    #    export_filter_calibration_button = QPushButton(
                    #        'Export Filter Calibration')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  export_filter_calibration)

                    #grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Export Serialization For SLAM
                    b = AppState.CalibrationActions.\
                        export_slam_calibration_button = QPushButton(
                            'Export SLAM Calibration')
                    b.pressed.connect(AppState.CalibrationActions.
                                      export_slam_calibration)

                    grid.addWidget(b, row(+1), 0)
                    #
                    # QPushButton -> Export Serialization For AI
                    #b = AppState.CalibrationActions.\
                    #    export_ai_calibration_button = QPushButton(
                    #        'Export AI Calibration')
                    #b.pressed.connect(AppState.CalibrationActions.
                    #                  export_ai_calibration)

                    #grid.addWidget(b, row(+1), 0)

                    self.setLayout(grid)

                    p = QPalette()
                    p.setColor(self.backgroundRole(),
                               COLORS['calibration_widget_bg_color'])
                    self.setPalette(p)
                    self.setAutoFillBackground(True)

                    if AppState.properties['CalibrationActions']['Minimized']:
                        self.setVisible(False)
                    else:
                        self.setVisible(True)

            def __init__(self, title, width):
                super(QtWindow.UserInputWidget.CalibrationWidget,
                      self).__init__()
                self.width = width
                self.title = title
                self.__init_widget()

            def __init_widget(self):
                grid = QGridLayout()
                #grid.addWidget(QtWindow.UserInputWidget.CalibrationWidget.
                #               TitleWidget(self.title, self.width), 0, 0)

                self.setLayout(grid)
                p = QPalette()
                p.setColor(self.backgroundRole(),
                           COLORS['calibration_widget_bg_color'])
                self.setPalette(p)
                self.setAutoFillBackground(True)

        def __init__(self):
            super(QtWindow.UserInputWidget, self).__init__()
            grid = QGridLayout()
            #
            # Filter Calibration
            self.__filter_calibration = self.CalibrationWidget.\
                FilterCalibration(self)
            self.__filter_calibration_title = self.CalibrationWidget.\
                TitleWidget("Filter Calibration", self.__filter_calibration,
                            AppState.FilterCalibration.minimized,
                            not AppState.properties['FilterCalibration'][
                                'Minimized'])

            grid.addWidget(self.__filter_calibration_title, 0, 0)
            grid.addWidget(self.__filter_calibration, 1, 0)
            grid.addItem(QSpacerItem(2, 2), 2, 0)
            #
            # Camera Calibration
            self.__cam_calibration = self.CalibrationWidget.CamCalibration(self)
            self.__cam_calibration_title = self.CalibrationWidget.TitleWidget(
                "Camera Calibration", self.__cam_calibration,
                AppState.CameraCalibration.minimized,
                not AppState.properties['CameraCalibration']['Minimized'])

            grid.addWidget(self.__cam_calibration_title, 3, 0)
            grid.addWidget(self.__cam_calibration, 4, 0)
            grid.addItem(QSpacerItem(2, 2), 5, 0)
            #
            # Mapping Calibration
            self.__mapping_calibration = self.CalibrationWidget.\
                MappingCalibration(self)
            self.__mapping_calibration_title = self.CalibrationWidget.\
                TitleWidget('Mapping Calibration', self.__mapping_calibration,
                            AppState.MappingCalibration.minimized,
                            not AppState.properties['MappingCalibration'][
                                'Minimized'])
            grid.addWidget(self.__mapping_calibration_title, 6, 0)
            grid.addWidget(self.__mapping_calibration, 7, 0)
            grid.addItem(QSpacerItem(2, 2), 8, 0)
            #
            # Real World Calibration
            self.__real_world_calibration = self.CalibrationWidget.\
                RealWorldCalibration(self)
            self.__real_world_calibration_title = self.CalibrationWidget.\
                TitleWidget("Real World Calibration",
                            self.__real_world_calibration,
                            AppState.RealWorldCalibration.minimized,
                            not AppState.properties['RealWorldCalibration'][
                                'Minimized'])

            grid.addWidget(self.__real_world_calibration_title, 9, 0)
            grid.addWidget(self.__real_world_calibration, 10, 0)
            grid.addItem(QSpacerItem(2, 2), 11, 0)
            #
            # Camera Configuration
            self.__cam_configuration = self.CalibrationWidget.\
                CameraConfiguration(self)
            self.__cam_configuration_title = self.CalibrationWidget.TitleWidget(
                "Camera Configuration", self.__cam_configuration,
                AppState.CameraConfiguration.minimized,
                not AppState.properties['CameraConfiguration']['Minimized'])

            grid.addWidget(self.__cam_configuration_title, 12, 0)
            grid.addWidget(self.__cam_configuration, 13, 0)
            grid.addItem(QSpacerItem(2, 2), 14, 0)
            #
            # Element Views
            self.__calibration_views = self.CalibrationWidget.ElementViews(self)
            self.__element_views_title = self.CalibrationWidget.TitleWidget(
                "Element Views", self.__calibration_views,
                AppState.ElementViews.minimized,
                not AppState.properties['ElementViews']['Minimized'])

            grid.addWidget(self.__element_views_title, 15, 0)
            grid.addWidget(self.__calibration_views, 16, 0)
            grid.addItem(QSpacerItem(2, 2), 17, 0)
            #
            # Calibration Actions
            self.__calibration_actions = self.CalibrationWidget.\
                CalibrationActions(self)
            self.__calibration_actions_title = self.CalibrationWidget.\
                TitleWidget("Calibration Actions", self.__calibration_actions,
                            AppState.CalibrationActions.minimized,
                            not AppState.properties['CalibrationActions'][
                                'Minimized'])

            grid.addWidget(self.__calibration_actions_title, 18, 0)
            grid.addWidget(self.__calibration_actions, 19, 0)

            grid.setRowStretch(20, 1)
            grid.setColumnMinimumWidth(0, 172)
            grid.setSpacing(0)
            grid.setHorizontalSpacing(0)
            grid.setVerticalSpacing(0)
            grid.setContentsMargins(0, 0, 0, 0)
            self.setLayout(grid)
            self.setFont(QFont('Cantarell', 9))

        def select_camera_element(self, element_id):
            self.__cam_calibration.refresh_calibration()

    def __init__(self):
        super(QtWindow, self).__init__()
        self.__camera_widget = QtWindow.CameraWidget(
            self.__camera_element_selected)

        self.__user_input_widget = QtWindow.UserInputWidget()
        self.__init_window()

        self.__display_timer = QTimer(self)
        self.connect(self.__display_timer,
                     SIGNAL("timeout()"),
                     self.display_camera)
        self.__display_timer.start(100)

        self.__run_timer = True
        self.__processing_view = False

        AppState.main_window = self

    def __init_window(self):
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.__user_input_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.verticalScrollBar().setVisible(True)

        grid = QGridLayout()
        grid.addWidget(self.__camera_widget, 0, 0)
        grid.addWidget(scroll_area, 0, 1)

        grid.setColumnStretch(1, 1)
        grid.setSpacing(0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        self.setLayout(grid)

        p = QPalette()
        p.setColor(self.backgroundRole(), COLORS['window_bg_color'])
        self.setPalette(p)

        self.setAutoFillBackground(True)
        self.setGeometry(0, 10, 1596, 794)
        self.setMinimumSize(self.size())
        self.setWindowTitle("Camera Mapping Calibration")
        self.showNormal()

        self.__camera_widget.setFixedSize(self.__camera_widget.size())

    def __camera_element_selected(self, element_id):
        AppState.selected_element_index = element_id
        self.__user_input_widget.select_camera_element(element_id)

    def display_camera(self):
        # executed each 25 ms (see self.__display_timer)
        if self.__run_timer:
            self.__processing_view = True
            AppState.CameraCalibration.element_preview.repaint()
            self.__camera_widget.refresh_camera_widget()
            self.__processing_view = False

    def stop_timer(self):
        self.__run_timer = False
        while self.__processing_view:
            time.sleep(0.1)
        self.__display_timer.stop()


def start_calibration(cam):
    AppState.gpu_view_operations = GpuViewOperations(cam)
    AppState.load_pickle_files()
    AppState.application = QApplication(sys.argv)
    AppState.application.lastWindowClosed.connect(
        AppState.save_state_to_pickled_file)
    window = QtWindow()

    def oups():
        for e in AppState.elements:
            def degree_to_radian(degree_val):
                return op.mul(degree_val, op.truediv(math.pi, 180))
            e['RelativeDegree'] = abs(e['RelativeDegree'] - 270.0)
            e['RelativeRadianAngle'] = degree_to_radian(e['RelativeDegree'])

    oups()

    from ngv_filter import NGVStructure
    for e in NGVStructure.create_ordered_structure(AppState.elements):
        print e
    #test_trigo()
    #from ngv_filter import NGVStructure
    #for e in NGVStructure.create_ordered_structure(AppState.elements):
    #    print e

    sys.exit(AppState.application.exec_())


def testing():
    print 'hello world'

if __name__ == "__main__":

    NGVFilter.start_calibration_gui(camera.Camera())
    testing()