#!/usr/bin/env python
import csv
import sys

w=50
h=50

if len(sys.argv) == 1:
    print('Usage: {} CSVFILE'.format(sys.argv[0]))
    exit(1)

with open(sys.argv[1], 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = [e for e in reader][1:]

    print('<svg width="{}" height="{}" viewbox="0 0 800 800" xmlns="http://www.w3.org/2000/svg">'.format(w*16, h*16))
    hidx=0
    xidx=0
    yidx=0
    for y in range(16):
        for x in range(16):
            idx = y*16+x
            print('  <rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})"/>'.format(w*xidx,h*yidx,w,h,data[idx][1],data[idx][2],data[idx][3]))
            if y == 15 and x == 15:
                break
            xidx -= 1
            yidx += 1
            if xidx < 0 or yidx >= 16:
                hidx += 1
                xidx = hidx
                yidx = 0
                while xidx >= 16:
                    xidx -= 1
                    yidx += 1
    print('</svg>')
