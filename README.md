Create Color Maps
=================

This is a Python module to create color maps with pretty much arbitrary size
according to the work of Kenneth Moreland as described at

(http://www.kennethmoreland.com/color-maps/)[http://www.kennethmoreland.com/color-maps/]

These color maps allow mapping a scalar value from a finite range to a color.  This is useful
in heatmaps or space-filling curves.

Objects with the predefined parameters can be created using the `make_*` functions
like this:

     map = moreland.make_kindlmann()

These objects are for now meant to
- create a SVG file with the color map, usable to show the range
- CSV files with the RGB color values, selectable in saturated (0-255) or real (0-1.0) form, or both


Create Individual Color Map File
--------------------------------

Use the `get-color-map.py` script to get individual CSV files describing the selected color map:

     python3 get-color-map.py viridis 256 byte

This will create the saturaged color Viridis map with 256 entries.  It is also possible to create the
real version of the map by using `float` instead `byte`.  Or use both.

The output of the command above is in the file `viridis-table-byte-0256.csv` which starts/ends with the
following lines:

     scalar,RGB_r,RGB_g,RGB_b
     0.0,72,0,84
     0.00392156862745098,73,0,86
     0.00784313725490196,73,0,87
     0.011764705882352941,74,0,89
     0.01568627450980392,74,1,90
     ...
     0.9882352941176471,236,232,21
     0.9921568627450981,238,232,24
     0.996078431372549,241,232,26
     1.0,243,233,28

The first value is the scalar value, ranging from 0 to 1.  By mapping the range [A,B] of the values which
are meant to be represented as color to the range [0,1] one selects the color with the scalar value closest
to the mapped value.  The values are equal distance which means that the scalar value in the first column
does not actually have to be used, the index into the table can for a value `v` can be computed with
`(v-A)/(B-A)`.

The color values are in this case given as a triple of bytes.  If the last parameter to the `get-color-maps.py`
script is `float` instead of `byte` the color is given as a triple of floating-point values.


Requirements
------------

On Fedora systems not all used Python packages are packaged as of Fedora 33.  Explicitly install the
packages first with

     pip install colormath toyplot
