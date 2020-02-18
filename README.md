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


Create Individual Color Map File
--------------------------------

Use the `get-color-map.py` script to get individual CSV files describing the selected color map:

     python3 get-color-map.py viridis 256 byte

This will create the saturaged color Viridis map with 256 entries.  It is also possible to create the
real version of the map by using `float` instead `byte`.  Or use both.
