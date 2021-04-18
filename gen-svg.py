#!/usr/bin/env python
import csv
import math
import argparse
import sys

parser = argparse.ArgumentParser(description='Visualize color maps.')
parser.add_argument('--fine', dest='fine', action='store_const',
                    const=True, default=False,
                    help='granularity of the image (default: coarse)')
parser.add_argument('csvfile', nargs=1, help='CSV file with colors')

args = parser.parse_args()

with open(args.csvfile[0], 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = [e for e in reader][1:]

    if args.fine:
        w=4
        h=4
        ncolors=len(data)
        s = ncolors // 2

        print('<svg width="{}" height="{}" viewbox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">'.format(w*s, h*s, w*s, h*s))
        for x in range(ncolors):
            xidx=x
            yidx=0
            while xidx >= s:
                xidx -= 1
                yidx += 1
            while xidx >= 0 and yidx < s:
                print('  <rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})"/>'.format(w*xidx,h*yidx,w,h,data[x][1],data[x][2],data[x][3]))
                xidx -= 1
                yidx += 1
        print('</svg>')
    else:
        w=50
        h=50
        s=16

        print('<svg width="{}" height="{}" viewbox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg">'.format(w*s, h*s, w*s, h*s))
        hidx=0
        xidx=0
        yidx=0
        for y in range(s):
            for x in range(s):
                idx = y*s+x
                print('  <rect x="{}" y="{}" width="{}" height="{}" fill="rgb({},{},{})"/>'.format(w*xidx,h*yidx,w,h,data[idx][1],data[idx][2],data[idx][3]))
                if y == s-1 and x == s-1:
                    break
                xidx -= 1
                yidx += 1
                if xidx < 0 or yidx >= s:
                    hidx += 1
                    xidx = hidx
                    yidx = 0
                    while xidx >= s:
                        xidx -= 1
                        yidx += 1
        print('</svg>')
