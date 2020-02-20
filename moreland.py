#
# This code is mostly from Kenneth Moreland's Jupyter notebooks at
#
#   https://github.com/kennethmoreland-com/kennethmoreland-com.github.io/tree/master/color-advice
#
# This version combines the different notebooks into one file and creates classes
# for the individual color map type.  The classes can be used but in many cases
# it can be sufficient to use the `make_*' to create pre-defined color models
# according to Moreland's document.
#
from __future__ import print_function

from colormath.color_objects import *
from colormath.color_conversions import convert_color, color_conversion_function
from colormath import color_diff

import toyplot
import toyplot.svg
import pandas
import numpy
import json


class _MshColor(IlluminantMixin, ColorBase):
    '''
    Represents an Msh color as defined by [Moreland2009]. The Msh color space
    is basically just a polar representation of CIE Lab space.
    See `Diverging Color Maps for Scientific Visualization
    <http://www.kennethmoreland.com/color-maps/>` for more information.
    '''

    VALUES = ['msh_m', 'msh_s', 'msh_h']

    def __init__(self, msh_m, msh_s, msh_h, observer = '2', illuminant = 'd50'):
        """
        :param float msh_m: M coordinate.
        :param float msh_s: s coordinate.
        :param float msh_h: h coordinate.
        :keyword str observer: Observer angle. Either ```2``` or ```10``` degrees.
        :keyword str illuminant: See :doc:`illuminants` for valid values.
        """

        super(_MshColor, self).__init__()
        #: M coordinate
        self.msh_m = float(msh_m)
        #: C coordinate
        self.msh_s = float(msh_s)
        #: H coordinate
        self.msh_h = float(msh_h)

        #: The color's observer angle. Set with :py:meth:`set_observer`.
        self.observer = None
        #: The color's illuminant. Set with :py:meth:`set_illuminant`.
        self.illuminant = None

        self.set_observer(observer)
        self.set_illuminant(illuminant)

@color_conversion_function(LabColor, _MshColor)
def _Lab_to_Msh(cobj, *args, **kwargs):
    """
    Convert from CIE Lab to Msh.
    """
    msh_m = math.sqrt(math.pow(float(cobj.lab_l),2) +
                      math.pow(float(cobj.lab_a),2) +
                      math.pow(float(cobj.lab_b),2))
    msh_s = math.acos(float(cobj.lab_l) / msh_m)
    msh_h = math.atan2(float(cobj.lab_b), float(cobj.lab_a))

    return _MshColor(msh_m, msh_s, msh_h, observer = cobj.observer, illuminant = cobj.illuminant)

@color_conversion_function(_MshColor, LabColor)
def _Msh_to_Lab(cobj, *args, **kwargs):
    """
    Convert from Msh to Lab.
    """
    lab_l = cobj.msh_m * math.cos(float(cobj.msh_s))
    lab_a = cobj.msh_m * math.sin(float(cobj.msh_s)) * math.cos(float(cobj.msh_h))
    lab_b = cobj.msh_m * math.sin(float(cobj.msh_s)) * math.sin(float(cobj.msh_h))
    return LabColor(lab_l, lab_a, lab_b, illuminant = cobj.illuminant, observer = cobj.observer)


def _safe_color(color):
    '''Given a color from the colormath.color_objects package,
    returns whether it is in the RGB color gamut and far enough
    away from the gamut border to be considered 'safe.' Colors
    right on the edge of displayable colors sometimes do not
    display quite right and also sometimes leave the color
    gamut when interpolated.'''
    rgb_color = convert_color(color, sRGBColor)
    rgb_vector = rgb_color.get_value_tuple()
    clamp_dist = 0.05 * (numpy.max(rgb_vector) - numpy.min(rgb_vector))
    return (rgb_color.rgb_r >= clamp_dist and rgb_color.rgb_r <= 1 - clamp_dist and
            rgb_color.rgb_g >= clamp_dist and rgb_color.rgb_g <= 1 - clamp_dist and
            rgb_color.rgb_b >= clamp_dist and rgb_color.rgb_b <= 1 - clamp_dist)



def _unzip_rgb_triple(dataframe, column = 'RGB'):
    '''Given a dataframe and the name of a column holding an RGB triplet,
    this function creates new separate columns for the R, G, and B values
    with the same name as the original with '_r', '_g', and '_b' appended.'''
    # Creates a data frame with separate columns for the triples in the given column
    unzipped_rgb = pandas.DataFrame(dataframe[column].values.tolist(), columns = ['r', 'g', 'b'])
    # Add the columns to the original data frame
    dataframe[column + '_r'] = unzipped_rgb['r']
    dataframe[column + '_g'] = unzipped_rgb['g']
    dataframe[column + '_b'] = unzipped_rgb['b']



class _ColorMap:
    def __init__(self, name = None):
        self.name = name

    def plot(self, fname = None, labels = False):
        # Build arrays of scalars and colors
        scalar_array = pandas.Series(numpy.linspace(0, 1, 1024))
        sRGB_array = pandas.Series(self.map_scalar_array(scalar_array))
        rgb_array = sRGB_array.apply(lambda color: color.get_value_tuple())

        # Create toyplot colormap object
        palette = toyplot.color.Palette(colors = rgb_array.values)
        colormap = toyplot.color.LinearMap(palette = palette, domain_min = 0, domain_max = 1)

        # Create toyplot display of colors.
        width = 30
        canvas = toyplot.Canvas(width = (100 + width) if labels else width, height = 300)
        numberline = canvas.numberline(x1 = 16, x2 = 16, y1 = -7, y2 = 7)
        numberline.padding = 30
        numberline.axis.spine.show = False
        numberline.colormap(colormap, width = width, style = {'stroke':'lightgrey'})

        control_point_scalars = numpy.linspace(0, 1, 9)
        control_point_labels = []
        for scalar in control_point_scalars:
            control_point_labels.append(
                '{:1.1f}, {}'.format(
                    scalar,
                    self.map_scalar(scalar).get_upscaled_value_tuple()))

        numberline.axis.ticks.locator = toyplot.locator.Explicit(locations = control_point_scalars, labels = control_point_labels)
        numberline.axis.ticks.labels.angle = -90
        numberline.axis.ticks.labels.style = {'text-anchor':'start',
                                              'baseline-shift':'0%',
                                              '-toyplot-anchor-shift':'-15px'}

        toyplot.svg.render(canvas, (fname if fname else self.name) + '.svg')


    def get_table(self, length, byte = True, real = False):
        table = pandas.DataFrame({'scalar': numpy.linspace(0, 1, length)})
        table['sRGBColor'] = self.map_scalar_array(table['scalar'])
        if byte:
            table['RGB'] = table['sRGBColor'].apply(lambda rgb: rgb.get_upscaled_value_tuple())
        if real:
            table['sRGB'] = table['sRGBColor'].apply(lambda rgb: rgb.get_value_tuple())
        return table

    def write_table(self, length, basename = None, byte = True, real = True):
        color_table = self.get_table(length, byte, real)
        if byte:
            _unzip_rgb_triple(color_table, 'RGB')
            color_table.to_csv(((basename if basename else self.name) + '-table-byte-{:04}.csv').format(length),
                               index=False,
                               columns=['scalar', 'RGB_r', 'RGB_g', 'RGB_b'])
        if real:
            _unzip_rgb_triple(color_table, 'sRGB')
            color_table.to_csv(((basename if basename else self.name) + '-table-float-{:04}.csv').format(length),
                               index=False,
                               columns=['scalar', 'sRGB_r', 'sRGB_g', 'sRGB_b'],
                               header=['scalar', 'RGB_r', 'RGB_g', 'RGB_b'])

    def write_tables(self, basename = None, byte=True, real=True):
        for num_bits in range(3, 11):
            self.write_table(2 ** num_bits, basename = basename, byte = byte, real = real)


class SmoothDivergingColorMap(_ColorMap):
    def __init__(self,
                 low_color = sRGBColor(0.230, 0.299, 0.754),
                 high_color = sRGBColor(0.706, 0.016, 0.150),
                 mid_color = _MshColor(88.0, 0.0, 0.0),
                 name = None):
        """
        :param color low_color: The color at the low end of the map.
        :param color high_color: The color at the high end of the map.
        :param color mid_color: The color at the middle of the map. Should be unsaturated.
        """
        super(SmoothDivergingColorMap,self).__init__(name)
        self.low_msh = convert_color(low_color, _MshColor)
        self.high_msh = convert_color(high_color, _MshColor)

        # If the points are saturated and distinct, then we place a white point
        # in the middle. Otherwise we ignore it.
        if self.low_msh.msh_s > 0.05:
            if self.high_msh.msh_s > 0.05:
                if (abs(self.low_msh.msh_h - self.high_msh.msh_h) > math.pi / 3.0) \
                     and mid_color:
                    # Both endpoints are saturated and unique and a midpoint was
                    # given. Interpolate through this midpoint and compute an
                    # appropriate hue spin.
                    mid_msh = convert_color(mid_color, _MshColor)
                    self.midpoint_magnitude = mid_msh.msh_m
                    self.midpoint_low_hue = self.compute_hue_spin(self.low_msh,mid_msh)
                    self.midpoint_high_hue = self.compute_hue_spin(self.high_msh,mid_msh)
                else:
                    # Both endpoints are distinct colors, but they are either very close
                    # in hue or no middle point was given. In this case, interpolate
                    # directly between them.
                    self.midpoint_magnitude = None
            else:
                # The low color is saturated but the high color is unsaturated.
                # Interpolate directly between them, but adjust the hue of the unsaturated
                # high color.
                self.midpoint_magnitude = None
                self.high_msh.msh_h = self.compute_hue_spin(self.low_msh, self.high_msh)
        else:
            # The low color is unsaturated. Assume the high color is saturated. (If not,
            # then this is a boring map no matter what we do.) Interpolate directly
            # between them, but adjust the hue of the unsaturated low color.
            self.midpoint_magnitude = None
            self.low_msh.msh_h = self.compute_hue_spin(self.high_msh, self.low_msh)

    def compute_hue_spin(self, MshSaturated, MshUnsaturated):
        '''
        Given a saturated color and unsaturated color, both as MshColor objects,
        computes a spin component to use during interpolation in Msh space. The spin
        is considered the target hue to interpolate to.
        '''
        if MshSaturated.msh_m >= MshUnsaturated.msh_m:
            return MshSaturated.msh_h
        else:
            hSpin = (MshSaturated.msh_s *
                     math.sqrt(math.pow(MshUnsaturated.msh_m,2) -
                               math.pow(MshSaturated.msh_m,2)) /
                     (MshSaturated.msh_m * math.sin(MshSaturated.msh_s)))
            if MshSaturated.msh_h > -math.pi / 3:
                return MshSaturated.msh_h + hSpin
            else:
                return MshSaturated.msh_h - hSpin

    def print_self(self):
        print('Low Color:')
        print('\t', self.low_msh)
        print('\t', convert_color(self.low_msh, LabColor))
        print('\t', convert_color(self.low_msh, sRGBColor))

        print('Middle Color:')
        if (self.midpoint_magnitude):
            print('\t Magnitude', self.midpoint_magnitude)
            print('\t Low Hue', self.midpoint_low_hue)
            print('\t High Hue', self.midpoint_high_hue)
        else:
            print('\t No Midpoint')

        print('High Color:')
        print('\t', self.high_msh)
        print('\t', convert_color(self.high_msh, LabColor))
        print('\t', convert_color(self.high_msh, sRGBColor))

    def map_scalar(self, scalar, space=_MshColor):
        '''
        Given a scalar value between 0 and 1, map to a color. The color is
        returned as a sRGBColor object.

        :param float scalar: The value to map to a color.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        if scalar < 0:
            return convert_color(self.low_msh, sRGBColor)
        if scalar > 1:
            return convert_color(self.high_msh, sRGBColor)

        interp = scalar
        low_color = convert_color(self.low_msh, space)
        high_color = convert_color(self.high_msh, space)
        if self.midpoint_magnitude:
            # Adjust the interpolation around the midpoint
            if scalar < 0.5:
                interp = 2*scalar
                high_msh = _MshColor(self.midpoint_magnitude, 0, self.midpoint_low_hue)
                high_color = convert_color(high_msh, space)
            else:
                interp = 2*scalar - 1
                low_msh = _MshColor(self.midpoint_magnitude, 0, self.midpoint_high_hue)
                low_color = convert_color(low_msh, space)
        low_color = numpy.array(low_color.get_value_tuple())
        high_color = numpy.array(high_color.get_value_tuple())

        mid_color = interp*(high_color-low_color) + low_color
        rgb = convert_color(space(mid_color[0], mid_color[1], mid_color[2]), sRGBColor)

        if (rgb.rgb_r < -0.0019 or rgb.rgb_r > 1.0019 or
            rgb.rgb_g < -0.0019 or rgb.rgb_g > 1.0019 or
            rgb.rgb_b < -0.0019 or rgb.rgb_b > 1.0019):
            print('WARNING: Value at scalar %1.4f is out of range' % scalar, rgb.get_value_tuple())

        # Just in case the color leaves the color gammut, clamp to valid values.
        return sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b)

    def map_scalar_array(self, scalar_array, space=_MshColor):
        '''
        Given an array of scalar values between 0 and 1, map them to colors.
        The color is returned as a sRGBColor object.

        :param float scalar_array: Array of values to map to colors.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        f = numpy.vectorize(lambda x: self.map_scalar(x, space))
        return f(scalar_array)


class BentDivergingColorMap(_ColorMap):
    def __init__(self,
                 low_color = sRGBColor(0.230, 0.299, 0.754),
                 high_color = sRGBColor(0.706, 0.016, 0.150),
                 mid_color = LCHabColor(88.0, 0.0, 0.0),
                 low_hue_spin = -32,
                 high_hue_spin = 32,
                 name = None):
        """
        :param color low_color: The color at the low end of the map.
        :param color high_color: The color at the high end of the map.
        :param color mid_color: The color at the middle of the map. Should be unsaturated.
        :param float low_hue_spin: The amount of spin to put on the low side of the map (in degrees)
        :param float high_hue_spin: The amount of spin to put on the high side of the map (in degrees)
        """
        super(BentDivergingColorMap,self).__init__(name)
        self.low_lch = convert_color(low_color, LCHabColor)
        self.high_lch = convert_color(high_color, LCHabColor)

        # If the points are saturated and distinct, then we place a white point
        # in the middle. Otherwise we ignore it.
        if self.low_lch.lch_c > 5:
            if self.high_lch.lch_c > 5:
                if (abs(self.low_lch.lch_h - self.high_lch.lch_h) > 60.0) \
                     and mid_color:
                    # Both endpoints are saturated and unique and a midpoint was
                    # given. Interpolate through this midpoint and compute an
                    # appropriate hue spin.
                    mid_lch = convert_color(mid_color, LCHabColor)
                    self.midpoint_luminance = mid_lch.lch_l
                    self.midpoint_low_hue = self.low_lch.lch_h + low_hue_spin
                    self.midpoint_high_hue = self.high_lch.lch_h + high_hue_spin
                else:
                    # Both endpoints are distinct colors, but they are either very close
                    # in hue or no middle point was given. In this case, interpolate
                    # directly between them.
                    self.midpoint_luminance = None
            else:
                # The low color is saturated but the high color is unsaturated.
                # Interpolate directly between them, but adjust the hue of the unsaturated
                # high color.
                self.midpoint_luminance = None
                self.high_lch.lch_h = self.low_lch.lch_h + low_hue_spin
        else:
            # The low color is unsaturated. Assume the high color is saturated. (If not,
            # then this is a boring map no matter what we do.) Interpolate directly
            # between them, but adjust the hue of the unsaturated low color.
            self.midpoint_luminance = None
            self.low_lch.lch_h = self.high_lch.lch_h + high_hue_spin

    def print_self(self):
        print('Low Color:')
        print('\t', self.low_lch)
        print('\t', convert_color(self.low_lch, LabColor))
        print('\t', convert_color(self.low_lch, sRGBColor))

        print('Middle Color:')
        if (self.midpoint_luminance):
            print('\t Luminance', self.midpoint_luminance)
            print('\t Low Hue', self.midpoint_low_hue)
            print('\t High Hue', self.midpoint_high_hue)
        else:
            print('\t No Midpoint')

        print('High Color:')
        print('\t', self.high_lch)
        print('\t', convert_color(self.high_lch, LabColor))
        print('\t', convert_color(self.high_lch, sRGBColor))

    def map_scalar(self, scalar, space=LCHabColor):
        '''
        Given a scalar value between 0 and 1, map to a color. The color is
        returned as a sRGBColor object.

        :param float scalar: The value to map to a color.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        if scalar < 0:
            return convert_color(self.low_lch, sRGBColor)
        if scalar > 1:
            return convert_color(self.high_lch, sRGBColor)

        interp = scalar
        low_color = convert_color(self.low_lch, space)
        high_color = convert_color(self.high_lch, space)
        if self.midpoint_luminance:
            # Adjust the interpolation around the midpoint
            if scalar < 0.5:
                interp = 2 * scalar
                high_lch = LCHabColor(self.midpoint_luminance, 0, self.midpoint_low_hue)
                high_color = convert_color(high_lch, space)
            else:
                interp = 2 * scalar - 1
                low_lch = LCHabColor(self.midpoint_luminance, 0, self.midpoint_high_hue)
                low_color = convert_color(low_lch, space)
        low_color = numpy.array(low_color.get_value_tuple())
        high_color = numpy.array(high_color.get_value_tuple())

        mid_color = interp*(high_color-low_color) + low_color
        rgb = convert_color(space(mid_color[0], mid_color[1], mid_color[2]), sRGBColor)

        if (rgb.rgb_r < -0.0019 or rgb.rgb_r > 1.0019 or
            rgb.rgb_g < -0.0019 or rgb.rgb_g > 1.0019 or
            rgb.rgb_b < -0.0019 or rgb.rgb_b > 1.0019):
            print('WARNING: Value at scalar %1.4f is out of range' % scalar, rgb.get_value_tuple())

        return rgb

    def map_scalar_array(self, scalar_array, space=LCHabColor):
        '''
        Given an array of scalar values between 0 and 1, map them to colors.
        The color is returned as a sRGBColor object.

        :param float scalar_array: Array of values to map to colors.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        f = numpy.vectorize(lambda x: self.map_scalar(x, space))
        return f(scalar_array)


class _InterpolateMap(_ColorMap):
    def __init__(self, name = None):
        super(_InterpolateMap,self).__init__(name)

    def color_lookup_sRGBColor(self, scalar):
        if scalar < 0:
            return sRGBColor(0, 0, 0)
        for index in range(self.data.index.size - 1):
            low_scalar = self.data['scalar'][index]
            high_scalar = self.data['scalar'][index + 1]
            if scalar <= high_scalar:
                low_lab = self.data['lab_values'][index]
                high_lab = self.data['lab_values'][index + 1]
                interp = (scalar - low_scalar) / (high_scalar - low_scalar)
                mid_lab = LabColor(interp * (high_lab.lab_l - low_lab.lab_l) + low_lab.lab_l,
                                   interp * (high_lab.lab_a - low_lab.lab_a) + low_lab.lab_a,
                                   interp * (high_lab.lab_b - low_lab.lab_b) + low_lab.lab_b)
                # Just in case the color leaves the color gammut, clamp to valid values.
                # This should not happen, but can just from converting to and from LAB
                interp_color = convert_color(mid_lab, sRGBColor)
                return sRGBColor(interp_color.clamped_rgb_r,
                                 interp_color.clamped_rgb_g,
                                 interp_color.clamped_rgb_b)
        return sRGBColor(1, 1, 1)

    def print_self(self):
        pass

    def map_scalar(self, scalar):
        '''
        Given a scalar value between 0 and 1, map to a color. The color is
        returned as a sRGBColor object.

        :param float scalar: The value to map to a color.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        return self.color_lookup_sRGBColor(scalar)

    def map_scalar_array(self, scalar_array):
        '''
        Given an array of scalar values between 0 and 1, map them to colors.
        The color is returned as a sRGBColor object.

        :param float scalar_array: Array of values to map to colors.
        :param color_object space: The colormath color object to do interpolation in.
        '''
        f = numpy.vectorize(lambda x: self.map_scalar(x))
        return f(scalar_array)


class FileColorMap(_InterpolateMap):
    def __init__(self, fname, name = None):
        super(FileColorMap,self).__init__(name)
        with open(fname, 'r') as fd:
            raw_color_data = json.load(fd)[0]
            scalar = []
            rgb_values = []
            for i in range(0, len(raw_color_data['RGBPoints']), 4):
                scalar.append(raw_color_data['RGBPoints'][i + 0])
                rgb_values.append(sRGBColor(
                    raw_color_data['RGBPoints'][i + 1],
                    raw_color_data['RGBPoints'][i + 2],
                    raw_color_data['RGBPoints'][i + 3]
                ))
            self.data = pandas.DataFrame({'scalar': scalar, 'rgb_values': rgb_values})
            self.data['lab_values'] = self.data['rgb_values'].apply(lambda rgb: convert_color(rgb, LabColor))


class SampleColorMap(_InterpolateMap):
    def __init__(self, samples, name = None):
        super(SampleColorMap,self).__init__(name)
        self.data = pandas.DataFrame({'RGB': samples})
        self.data['rgb_values'] = self.data['RGB'].apply(lambda rgb: sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True))
        self.data['lab_values'] = self.data['rgb_values'].apply(lambda rgb: convert_color(rgb, LabColor))
        self.data['scalar'] = self.data['lab_values'].apply(lambda lab: lab.lab_l / 100.0)


class HueColorMap(_InterpolateMap):
    def __init__(self, starthue = 300.0, endhue = 0.0, name = None):
        super(HueColorMap,self).__init__(name)
        self.starthue = starthue
        self.diffhue = endhue - starthue

    def color_lookup_sRGBColor(self, scalar):
        hue = self.starthue + scalar * self.diffhue
        '''Given a hue value (in degrees) and a scalar value between
        0 and 1, create a color to have a luminance proportional to
        the scalar with the given hue. Returns an sRGBColor value.'''
        if scalar <= 0:
            return sRGBColor(0, 0, 0)
        if scalar >= 1:
            return sRGBColor(1, 1, 1)
        hsv_original = HSVColor(hue, 1.0, 1.0)
        rgb_original = convert_color(hsv_original, sRGBColor)
        lab_original = convert_color(rgb_original, LabColor)
        l_target = 100.0 * scalar
        a_original = lab_original.lab_a
        b_original = lab_original.lab_b

        high_scale = 1.0
        low_scale = 0.0
        for i in range(12):
            mid_scale = (high_scale - low_scale) / 2 + low_scale
            if _safe_color(LabColor(l_target, mid_scale * a_original, mid_scale * b_original)):
                low_scale = mid_scale
            else:
                high_scale = mid_scale

        return convert_color(LabColor(l_target, low_scale * a_original, low_scale * b_original), sRGBColor)


def make_smooth_diverging(name = 'smooth-cool-warm'):
    return SmoothDivergingColorMap(low_color = sRGBColor(0.230, 0.299, 0.754),
                                   high_color = sRGBColor(0.706, 0.016, 0.150),
                                   mid_color = sRGBColor(0.8654, 0.8654, 0.8654),
                                   name = name)

def make_bent_diverging(name = 'bent-cool-warm'):
    return BentDivergingColorMap(low_color = sRGBColor(0.230, 0.299, 0.754),
                                 high_color = sRGBColor(0.706, 0.016, 0.150),
                                 mid_color = sRGBColor(0.95, 0.95, 0.95),
                                 name = name)

def make_viridis(name = 'viridis'):
    return FileColorMap(fname = 'viridis-original.json', name = name)

def make_plasma(name = 'plasma'):
    return FileColorMap(fname = 'plasma-original.json', name = name)

def make_black_body(name = 'black-body'):
    return SampleColorMap([(0,0,0),          # black
                           (178,34,34),      # red
                           (227,105,5),      # orange
                           (238,210,20),     # yellow
                           (255, 255, 255)], # white
                           name = name)

def make_inferno(name = 'inferno'):
    return FileColorMap(fname = 'inferno-original.json', name = name)

def make_kindlmann(name = 'kindlmann'):
    return HueColorMap(starthue = 300.0, endhue = 0.0, name = name)

def make_extended_kindlmann(name = 'extended-kindlmann'):
    return HueColorMap(starthue = 300.0, endhue = -180.0, name = name)
