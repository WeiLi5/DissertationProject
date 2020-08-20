from pathlib import Path
from image import ImageLoader, Regrid
from PIL import Image
import numpy as numpy
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import math
import time
from skimage.restoration import inpaint
from skimage import io
from time import sleep
import os
import argparse

#For naming cropped images
product_name = ""
name = ""
radiances_folder = "RadianceBlocks/"
clouds_mask_folder = "CloudMaskBlocks/"
origin_rad_folder = "original_radiance/"
origin_mask_folder = "original_mask/"

def read_product(A,B):
	""" Fixture to return the path to the local testing product """
	global product_name
	global clouds_mask_folder

	paths = Path(A).glob('S3A*')
	for path in paths:
		product_name = path.name
		path_in_str = str(path)

		#Read a single product
		product = ImageLoader(path_in_str)
		#Check if the product is at day time
		flags = product.load_flags()
		is_day = numpy.all(flags.day.values == 1)
		if(is_day == True):
			print("Processing Product: ",product_name)
			visualize_natural_color_max(product,B)
			# visualize_natural_color_bi(product,B)
			read_bayes_in(product,B)
			# read_summary_mask(product,B)
			


		#read_radiances(product)
		#read_bayes_in(product)
		#visualize_natural_color(product)


def read_radiances(product):
	"""Read radiances channel"""

	rads = product.load_radiances()
	#save_image_in_actual_size(rads["S1_radiance_an"],"original_radiance_S1/")
	#save_image_in_actual_size(rads["S2_radiance_an"],"original_radiance_S2/")
	#save_image_in_actual_size(rads["S3_radiance_an"],"original_radiance_S3/")
	save_image_in_actual_size(rads["S4_radiance_an"],"original_radiance_S4/")
	# save_image_in_actual_size(rads["S5_radiance_an"],"original_radiance_S5/")
	# save_image_in_actual_size(rads["S6_radiance_an"],"original_radiance_S6/")

def read_bayes_in(product,B):
	global product_name
	"""Regrid bayes mask"""
	flags = product.load_flags()
	bayes = flags["bayes_in"]
	save_image_in_actual_size(bayes,B+"/"+"mask/")
	# img = Image.fromarray(bayes)
	# row,col = img.size
	# print(img.size)
	# imgR = img.resize((row*2,col*2))
	# print(imgR.size)
	# cropped_image = imgR.crop((100,5,row-80,col-5))
	# img.show()
	# print(cropped_image)
	# cropped_image.save("resize_mask/"+ product_name+".png")

def read_summary_mask(product,B):
	global product_name
	"""Regrid Summary mask"""
	flags = product.load_flags()
	bayes = flags["summary_cloud"]
	save_image_in_actual_size(bayes,B+"/")

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
	#img = img[5:r-5,100:c-80]
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


def resize_mask(input_path,output_path):
	"""Resize mask images"""
	paths = Path(input_path).glob('S3A*')
	for path in paths:
		resize_mask_name = path.name 
		print("Processing Image: ",resize_mask_name)
		path_in_str = str(path)
		im = Image.open(path).convert("L")
		width,height = im.size
		imR = im.resize((width*2,height*2),resample=Image.NEAREST)
		# print(imgR.size)
		# cropped_image = imgR.crop((100,5,row-80,col-5))
		row,col = imR.size
		imC = imR.crop((100,5,row-80,col-5))
		imC.save(output_path + "/"+ resize_mask_name)


def crop_image(input_folder,output_folder,w_size):
	paths = Path(input_folder).glob('S3A*')
	for path in paths:	
		image = Image.open(path)

		# Define the window size
		windowsize_r = w_size
		windowsize_c = w_size
		print("Processing image:",path.name)

		# Crop out the window and calculate the histogram
		for r in range(0,image.size[0] - windowsize_r, windowsize_r):
			for c in range(0,image.size[1] - windowsize_c, windowsize_c):
				cropped_image = image.crop((r,c,r+windowsize_r,c+windowsize_c))
				cropped_image.save(output_folder + "/" + str(r) + "_" + str(c) + "_" + path.name, "PNG", optimize=True)

def remove_bw_images(A,B):
	"""Remove the mask images with only black or white color
	Also remove the correspond radiance images"""
	paths = Path(A).glob('*')
	for path in paths:	
		image = Image.open(path)
		if(image.convert("L").getextrema() == (0,0)):
			print("black +++: " + path.name)
			os.remove(path)
			os.remove(B+'/'+path.name)
		elif(image.convert("L").getextrema() == (255,255)):
			print("white ----: " + path.name)
			os.remove(path)
			os.remove(B+'/'+path.name)

def visualize_natural_color_bi(product,B):
	#reference: https://stackoverflow.com/questions/42872293/channel-mix-with-pillow
	global product_name
	name = product_name + ".png"
	#read radiance channels
	# rads = product.load_radiances()
	# s1 = rads["S1_radiance_an"].values
	# s3 = rads["S3_radiance_an"].values
	# s5 = rads["S5_radiance_an"].values
	refs = product.load_reflectances()
	s1 = refs["S1_reflectance_an"].values*255
	s3 = refs["S3_reflectance_an"].values*255
	s5 = refs["S5_reflectance_an"].values*255

	#----------test--------------
	start_time = time.time()
	row,col = s1.shape
	s1 = s1[5:row-5,100:col-80]
	#fill s1 NaNs
	mask1 = numpy.isnan(s1)
	f_s1 = inpaint.inpaint_biharmonic(s1,mask1)
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	s3 = s3[5:row-5,100:col-80]
	#fill s3 NaNs
	mask3 = numpy.isnan(s3)
	f_s3 = inpaint.inpaint_biharmonic(s3,mask3)
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	s5 = s5[5:row-5,100:col-80]
	#fill s5 NaNs
	mask5 = numpy.isnan(s5)
	f_s5 = inpaint.inpaint_biharmonic(s5,mask5)
	print("--- %s seconds ---" % (time.time() - start_time))
	#-----------------------------

	# # Transform channel data
	r, g, b = (f_s3 + f_s5) / 2, f_s3, (f_s3 + f_s1) / 2

	# Merge channels
	out_data = numpy.stack((r, g, b), axis=2).astype('uint8')
	out_img = Image.fromarray(out_data)
	out_img.save(B + "/" +"origin/"+ name,"PNG", optimize=True)


def visualize_natural_color_max(product,B):
	global product_name
	# Save the image.
	name = product_name + ".png"
	#reference: https://stackoverflow.com/questions/42872293/channel-mix-with-pillow
	#read radiance channels

	# path = "data/S3A_SL_1_RBT____20200417T022539_20200417T022839_20200418T070654_0179_057_160_3060_LN2_O_NT_004.SEN3"
	# product = ImageLoader(path)
	# rads = product.load_radiances()
	# s1 = rads["S1_radiance_an"].values
	# s3 = rads["S3_radiance_an"].values
	# s5 = rads["S5_radiance_an"].values
	refs = product.load_reflectances()
	s1 = refs["S1_reflectance_an"].values*255
	s3 = refs["S3_reflectance_an"].values*255
	s5 = refs["S5_reflectance_an"].values*255

	#----------test--------------
	start_time = time.time()
	row,col = s1.shape
	s1 = s1[5:row-5,100:col-80]
	#fill s1 NaNs
	f_s1 = numpy.where(numpy.isnan(s1),s1[~numpy.isnan(s1)].max(),s1)
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	s3 = s3[5:row-5,100:col-80]
	#fill s3 NaNs
	mask3 = numpy.isnan(s3)
	f_s3 = numpy.where(numpy.isnan(s3),s3[~numpy.isnan(s3)].max(),s3)
	print("--- %s seconds ---" % (time.time() - start_time))

	start_time = time.time()
	s5 = s5[5:row-5,100:col-80]
	#fill s5 NaNs
	mask5 = numpy.isnan(s5)
	f_s5 = numpy.where(numpy.isnan(s5),s5[~numpy.isnan(s5)].max(),s5)
	print("--- %s seconds ---" % (time.time() - start_time))
	#-----------------------------

	# Transform channel data
	r, g, b = (f_s3 + f_s5) / 2, f_s3, (f_s3 + f_s1) / 2
	# Merge channels
	out_data = numpy.stack((r, g, b), axis=2).astype('uint8')
	out_img = Image.fromarray(out_data)
	out_img.save(B + "/" + name,"PNG", optimize=True)




def main():
	#find . -name ".DS_Store" -delete
	#python datasets/combine_A_and_B.py --fold_A datasets/aligned/A --fold_B datasets/aligned/B --fold_AB datasets/AtoB_lambda_AB
	# read_product()
	# resize_mask("summary_mask","summary_mask/resized_mask")
	# crop_image("resize_mask","mask_cropped/")
	#crop_image("natural_color","natural_cropped/")
	# crop_image("summary_mask/resized_mask","summary_mask/cropped_mask")
	#remove_bw_images("natural_color_pix2pix/A/train","natural_color_pix2pix/B/train/")
	# visualize_natural_color_bi()
	# visualize_natural_color_max()

	parser = argparse.ArgumentParser()
	parser.add_argument("--folder_A", type=str,help="input folder path")
	parser.add_argument("--folder_B", type=str,help="output folder path")
	parser.add_argument("--read_product", action="store_true",help="read products and output natual color images")
	parser.add_argument("--resize_mask", action="store_true",help="resize the masks to the same size of natural color images")
	parser.add_argument("--crop_image", action="store_true",help="crop the image into small blocks")
	parser.add_argument("--window_size",type=int,help="size")
	parser.add_argument("--remove_bw", action="store_true",help="crop the image into small blocks")


	args = parser.parse_args()

	if (args.read_product):
		#python3 slstr_preprocess.py --read_product --folder_A data --folder_B summary_mask
		read_product(args.folder_A,args.folder_B)
	elif(args.resize_mask):
		#python3 slstr_preprocess.py --resize_mask --folder_A summary_mask --folder_B summary_mask/resized_mask
		resize_mask(args.folder_A,args.folder_B)
	elif(args.crop_image):
		#python3 slstr_preprocess.py --crop_image --folder_A summary_mask/resized_mask --folder_B summary_mask/cropped_mask --window_size 256
		crop_image(args.folder_A,args.folder_B,args.window_size)
	elif(args.remove_bw):
		#python slstr_preprocess.py --remove_bw --folder_A test_results/reconstruction/A/test --folder_B test_results/reconstruction/B/test
		remove_bw_images(args.folder_A,args.folder_B)



	

if __name__== "__main__":
	main()

