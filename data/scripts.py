import sys, os
import urllib2
import numpy as np

def main():
	pwd = os.getcwd()
	path = os.path.join(pwd, "imageNet")
	im_list = os.listdir(path)

	for fname in im_list:
		fpath = os.path.join(path, fname)
		f = open(fpath, "r")

		f.close() 


if __name__ == "__main__":
	main()