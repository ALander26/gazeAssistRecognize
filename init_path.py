
"""Set up paths."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, 'py-fast-rcnn','caffe-fast-rcnn', 'python')
add_path(caffe_path)

# Add fast rcnnlib to PYTHONPATH
fast_rcnn_lib_path = osp.join(this_dir, 'py-fast-rcnn', 'lib')
add_path(fast_rcnn_lib_path)

# Add functions to PYTHONPATH
func_path = osp.join(this_dir, 'functions')
add_path(func_path)

# Add data to PYTHONPATH
data_path = osp.join(this_dir, 'data')
add_path(data_path)

# Add bing objectness
bing_path = osp.join(this_dir, 'lib', 'BING-Objectness', 'source')
add_path(bing_path)
# # Add external library path
# lib_path = osp.join(this_dir, 'libs', 'saliencyMap', 'src')
# add_path(lib_path)