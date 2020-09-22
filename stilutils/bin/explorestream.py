#!/usr/bin/env python

from __future__ import print_function
import os
import sys

try:
    input_stream = sys.argv[1]
except IndexError:
    print("Usage: python explorestream.py output.stream")
    exit(1)

if input_stream == "-h" or input_stream == "--help":
    print("Usage: python explorestream.py output.stream")

os.system("grep Cell %s > tmp.lst" % input_stream)
fin = [elem.split() for elem in open("tmp.lst").read().split("\n") if elem]

a = [float(elem[2]) * 10 for elem in fin]
b = [float(elem[3]) * 10 for elem in fin]
c = [float(elem[4]) * 10 for elem in fin]
al = [float(elem[6]) for elem in fin]
be = [float(elem[7]) for elem in fin]
ga = [float(elem[8]) for elem in fin]


def stats(lst):
    average = lambda lst: float(sum(lst)) / len(lst)
    mean = average(lst)
    sigma = average([(elem - mean) ** 2 for elem in lst]) ** 0.5
    return mean, sigma


print("Cell parameters for %s are:" % input_stream)
print("a = %.2f A\trmsd = %.2f A" % stats(a))
print("b = %.2f A\trmsd = %.2f A" % stats(b))
print("c = %.2f A\trmsd = %.2f A" % stats(c))
print("al= %.2f deg\trmsd = %.2f deg" % stats(al))
print("be= %.2f deg\trmsd = %.2f deg" % stats(be))
print("ga= %.2f deg\trmsd = %.2f deg" % stats(ga))

os.system("rm tmp.lst")
