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
from skimage.restoration import inpaint
from skimage import io
from time import sleep

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
		#read_radiances(product)
		read_bayes_in(product)


def read_radiances(product):
	"""Read radiances channel"""

	rads = product.load_radiances()
	#save_image_in_actual_size(rads["S1_radiance_an"],"original_radiance_S1/")
	#save_image_in_actual_size(rads["S2_radiance_an"],"original_radiance_S2/")
	#save_image_in_actual_size(rads["S3_radiance_an"],"original_radiance_S3/")
	save_image_in_actual_size(rads["S4_radiance_an"],"original_radiance_S4/")
	# save_image_in_actual_size(rads["S5_radiance_an"],"original_radiance_S5/")
	# save_image_in_actual_size(rads["S6_radiance_an"],"original_radiance_S6/")

def read_bayes_in(product):
	"""Regrid bayes mask"""
	flags = product.load_flags()
	save_image_in_actual_size(flags["bayes_in"],"original_mask/")

#reference: https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot
def save_image_in_actual_size(data_set,saving_folder):
	"""Takes a data set and save it as .png image"""
	global product_name
	global name
	sleep(0.01)
	#dpi = 80
	dpi = matplotlib.rcParams['figure.dpi']
	img = data_set.values
	r,c = img.shape
	img = img[5:r-5,100:c-80]
	#fill NaNs
	mask = numpy.isnan(img)
	start_time = time.time()
	print("Start processing: " + product_name + "--" + data_set.name)
	try:
		im_data = inpaint.inpaint_biharmonic(img,mask)
		print("--- %s seconds ---" % (time.time() - start_time))
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
		name = product_name + ".png"
		ax.imshow(im_data,cmap='gray')
		plt.savefig(saving_folder + name)
		plt.close()

	except ValueError:  #raised if xx is empty.
		print("++++++++ValueError in this product: "+ product_name+ "--" + data_set.name)
		pass


def resize_mask():
	"""Resize mask images"""
	paths = Path('original_mask').glob('S3A*')
	for path in paths:
		resize_mask_name = path.name 
		path_in_str = str(path)
		im = Image.open(path)
		imR = im.resize((3000,2400))
		imR.save("resize_mask/"+ resize_mask_name)


def crop_image(input_folder,output_folder):
	paths = Path(input_folder).glob('S3A*')
	for path in paths:	
		image = Image.open(path)

		# Define the window size
		windowsize_r = 256
		windowsize_c = 256

		# Crop out the window and calculate the histogram
		for r in range(0,image.size[0] - windowsize_r, windowsize_r):
			for c in range(0,image.size[1] - windowsize_c, windowsize_c):
				cropped_image = image.crop((r,c,r+windowsize_r,c+windowsize_c)).convert("LA")
				cropped_image.save(output_folder + str(r) + "_" + str(c) + "_" + path.name, "PNG", optimize=True)
def main():
	read_product()
	resize_mask()
	#crop_image("resize_mask","mask_cropped/")
	#crop_image("original_radiance_S3","radiance_cropped_S3/")


	

if __name__== "__main__":
	main()

