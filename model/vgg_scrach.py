from __future__ import absolute_import

import random
import os

# import torch
import time
import os.path as osp
import sys

os.mkdir(os.path.join(os.getcwd(), "home"))

# vgg model

# Current file location
file_location = __file__
print(f"current file location : {file_location}")

absolute_path = osp.abspath(file_location)
print(f"absolute path {absolute_path}")

dirname = osp.dirname(absolute_path)
print(dirname)
ok = sys.path.append(dirname)
print(ok)


# p = torch.tensor(5)
# arr = torch.tensor([1, 2, 3, 9, 4])
# print(p)
# print(p.item())
# a, b = torch.max(arr)
# print(a)

# ok = sys.path(os.path.dirname(os.path.abspath))
# dirpath = os.path.dirname
# print(dirpath)
# abspath = os.path.abspath
# print(abspath)
# print(__file__)
x: int = 10
if 5 < 6:
    print("Hello")
# x = "kkdk"
tuple1 = ("hhelll", "ik", "ij")
tuple2 = (1, 2, 3)
tuple_union = tuple1 + tuple2
print(tuple_union)
a, b, c = tuple1
print(a, b, c)
dict = {"ok": 1, "l": 2}
dict1 = {**dict, "ol": 5}
dict["ok"] = 38
print(dict)
print(dict1)
for i in range(len(tuple1)):
    print(tuple1[i])

for i, x in enumerate(tuple1):
    print(i, x)


os.removedirs(os.getcwd() + "/ok")
# print()

x = random.randint(100, 1000)

print(x)

# ll: list[str] = [1, 2, 3]


start = time.time()

end = time.time()

print(end - start)
