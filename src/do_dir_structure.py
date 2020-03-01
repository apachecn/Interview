#!/usr/bin/python3
# coding: utf8
import os
import sys
# 自定义包(添加：中间件)
sys.path.append(os.getcwd())
from Middleware.tool import get_dir_files

catalog = "docs/Algorithm/Leetcode/Python"
files_path = get_dir_files(catalog, [], status=-1, str1=".DS_Store")
# print(files_path)

for line in files_path:
    if "ipynb" not in line:
        l_file = line.split("/")[-2:]
        filename = "%s %s" % (l_file[-1].split(".")[0], l_file[-1].split(".")[1].replace("_", " ").strip())
        filepath = "    * [%s](%s)" % (filename, "/".join(l_file))
        # print(">>> ", filepath)
        print(filepath)