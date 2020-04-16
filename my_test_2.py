from pathlib import Path
from image import ImageLoader, Regrid
from PIL import Image
import numpy as numpy
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import math
import time

#For naming cropped images
product_name = ""
name = ""
radiances_folder = "RadianceBlocks/"
clouds_mask_folder = "CloudMaskBlocks/"
origin_rad_folder = "original_radiance/"
origin_mask_folder = "original_mask/"

def read_product():
	""" Fixture to return the path to the local testing product """
	global product_name
	global clouds_mask_folder

	paths = Path('data').glob('S3A*')
	for path in paths:
		product_name = path.name
		path_in_str = str(path)

		#Read a single product
		product = ImageLoader(path_in_str)
		fillNanTest(product)
		#read_bayes_in(product,origin_mask_folder)
		#read_radiances(product,origin_rad_folder)


def read_radiances(product,saving_folder):
	"""Read radiances channel"""

	rads = product.load_radiances()
	for name in rads:
		save_image_in_actual_size(rads[name],saving_folder)

def read_bayes_in(product,saving_folder):
	"""Regrid bayes mask"""
	flags = product.load_flags()
	save_image_in_actual_size(flags["bayes_in"],saving_folder)

#reference: https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
def save_image_in_actual_size(data_set,saving_folder):
	"""Takes a data set and save it as .png image"""
	global product_name
	global name

	#dpi = 80
	dpi = matplotlib.rcParams['figure.dpi']
	im_data = data_set
	height, width = im_data.shape
	print(im_data.shape)

	# What size does the figure need to be in inches to fit the image?
	figsize = width / float(dpi), height / float(dpi)

	# Create a figure of the right size with one axes that takes up the full figure
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0, 0, 1, 1])

	# Hide spines, ticks, etc.
	ax.axis('off')

	# Save the image.
	name = product_name + data_set.name + ".png"
	ax.imshow(im_data,cmap='gray')
	plt.savefig(saving_folder + name)
	plt.close()

def save_image_in_actual_size(data_set):
	"""Takes a data set and save it as .png image"""

	#dpi = 80
	dpi = matplotlib.rcParams['figure.dpi']
	im_data = data_set
	height, width = im_data.shape
	saving_folder = "fillNan/"
	# What size does the figure need to be in inches to fit the image?
	figsize = width / float(dpi), height / float(dpi)

	# Create a figure of the right size with one axes that takes up the full figure
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0, 0, 1, 1])

	# Hide spines, ticks, etc.
	ax.axis('off')

	# Save the image.
	ts = time.time()
	name = str(ts)+ ".png"
	ax.imshow(im_data,cmap='gray')
	plt.savefig(saving_folder + name)
	plt.close()

def resize_mask():
	paths = Path('original_mask').glob('S3A*')
	for path in paths:
		resize_mask_name = path.name 
		path_in_str = str(path)
		im = Image.open(path)
		imR = im.resize((3000,2400))
		imR.save("resize_mask/"+ resize_mask_name)

def fillNan():
	"""Cannot"""
	paths = Path('original_radiance').glob('S3A*')
	for path in paths:
		print(path)
		image = Image.open(path)
		r_image = image
		inds = numpy.where(numpy.isnan(r_image)) 
		r_image[inds] = numpy.take(0, inds[1]) 
		r_image.save("fillNan/"+ path.name)

def fillNanTest(product):
	rads = product.load_radiances()
	for name in rads:
		fn_rad = rads[name]
		result = numpy.where(numpy.isnan(fn_rad))
		print(result[0][1])
		for i in range (len(result[0])):
			row = result[0][i]
			col = result[1][i]
			mean = (fn_rad[row - 1][col] + fn_rad[row + 1][col] + fn_rad[row][col + 1] + fn_rad[col - 1])/4
		#fn_rad = numpy.where(numpy.isnan(fn_rad), 0, fn_rad)
		#save_image_in_actual_size(fn_rad)

def main():
	read_product()
	#resize_mask();
  
if __name__== "__main__":
	main()

