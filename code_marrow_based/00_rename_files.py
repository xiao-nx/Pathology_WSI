#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import shutil

def files_rename(files_path,new_name):
    dirs = os.listdir(files_path)
    for dir in dirs:
        src_path = os.path.join(files_path, dir)
        file_name = str(dir) + new_name
        dst_path = os.path.join(files_path, file_name)
        shutil.move(src_path, dst_path)

    return

if __name__ == '__main__':

    normal_path = '../dataSets/dataAnnotated/normal'  # 所需修改文件夹所在路径
    # infected_path = '../dataSets/dataAnnotated/unCLL/Infected/'
    files_rename(normal_path,'_normal')
    # files_rename(infected_path, '_infected')

