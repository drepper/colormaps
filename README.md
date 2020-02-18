Create Color Maps
=================

This is a Python module to create color maps with pretty much arbitrary size
according to the work of Kenneth Moreland as described at

[http://www.kennethmoreland.com/color-maps/]

Objects with the predefined parameters can be created using the `make_*` functions
like this:

     map = moreland.make_kindlmann()

These objects are for now meant to
- create a SVG file with the color map, usable to show the range
- CSV files with the RGB color values, selectable in saturated (0-255) or real (0-1.0) form, or both
