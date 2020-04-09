from pathlib import Path
from image import ImageLoader, Regrid
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import numpy as numpy

#For naming cropped images
product_name = ""
name = ""
radiances_folder = "RadianceBlocks/"
clouds_mask_folder = "CloudMaskBlocks/"


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

		#Read radiances and crop them into blocks
		read_radiances(product,radiances_folder)
		#regrid_radiances(product)

		read_bayes(product,clouds_mask_folder)



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
	ax.imshow(im_data,cmap='gray')
	plt.savefig(saving_folder + name)
	plt.close()

def crop_image(data_set,saving_folder):
	"""Crop an image into r*c blocks"""

	global product_name
	global name

	originImage = data_set

	# Define the window size
	windowsize_r = 256
	windowsize_c = 256

	# Crop out the window and calculate the histogram
	for r in range(0,originImage.shape[0] - windowsize_r, windowsize_r):
		for c in range(0,originImage.shape[1] - windowsize_c, windowsize_c):
			window = originImage[r:r+windowsize_r,c:c+windowsize_c]
			#create a name for the cropped image
			name = product_name + data_set.name + "_" + str(r) + "_" + str(c) + ".png"
			save_image_in_actual_size(window,saving_folder)

def read_radiances(product,saving_folder):
	"""Read radiances channel"""

	rads = product.load_radiances()
	for name in rads:
		crop_image(rads[name],saving_folder)

def read_bayes(product,saving_folder):
	"""Read bayes mask"""
	bayes_mask = regrid_bayes_in(product)
	crop_image(bayes_mask,saving_folder)
	# flags = product.load_flags()
	# bayes_mask = flags["bayes_in"]
	# crop_image(bayes_mask,saving_folder)


def regrid_bayes_in(product):
	"""Regrid bayes mask"""
	flags = product.load_flags()
	rads = product.load_radiances()
	regridder = Regrid(rads, flags)
	flags_resized = regridder(flags)
	return flags_resized['bayes_in']

	# #original mask
	# plt.imshow(flags['bayes_in'],cmap="gray")
	# plt.show()
	# print(flags['bayes_in'].shape)

def regrid_radiances(product):
	#Seems this is wrong method and casue errors
	flags = product.load_flags()
	rads = product.load_radiances()
	regridder = Regrid(flags,rads)
	radiances_resized = regridder(rads)
	plt.imshow(radiances_resized["S1_radiance_an"])
	plt.show()

	return radiances_resized


def read_flags():
	return true

def main():
	read_product()
  
if __name__== "__main__":
	main()
