from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
from pathlib import Path
import re
from shutil import copy2
from skimage.draw import random_shapes
from PIL import Image
from statistics import mean 
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from math import log10, sqrt 
from SSIM_PIL import compare_ssim

def load_image(infilename) :
	'''Load images as arrays'''
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

""" Used for caculating accuracy. Results will be printed into a txt file """
def caculate_accuracy(folder_A):
	out_file = open("acc_output.txt", "a")
	print("----------------------------------------folder:",folder_A,"---------------------------------------",file=out_file)
	#Get images paths
	real_paths = Path(folder_A).glob('*real_B.png')
	blank = 0
	product_dic = {}
	acc_list = []
	recall_list = []
	precision_list = []
	count = 0
	#open a txt file for recording

	for real_path in real_paths:
		#real_B path
		real_in_str = str(real_path)
		#fake_B path
		fake_path = re.sub('\.png$', '', real_in_str)
		fake_path = re.sub('real_B','',fake_path)
		fake_in_str = fake_path + "fake_B.png"

		#real_A path
		realA_path = re.sub('\.png$', '', real_in_str)
		realA_path = re.sub('real_B','',fake_path)
		realA_in_str = realA_path + "real_A.png"

		#product name
		product_name = re.search('S3A_(.*)SEN3',real_in_str).group(1)
		product_name = "S3A_" +product_name+ "SEN3"

		#read fake image
		fake_in_str = re.sub(folder_A+'/',folder_A+'/binary_fake/',fake_in_str)

		fake = Image.open(fake_in_str).convert("L")
		data_fake = np.asarray(fake)
		f = np.where(data_fake > 128,1,0)

		#read real image
		real = Image.open(real_in_str).convert("L")
		data_real = np.asarray(real)
		r = np.where(data_real > 128,1,0)

		#caculate confusion matrix
		r = r.flatten()
		f = f.flatten()
		try:
			tn, fp, fn, tp = confusion_matrix(r,f).ravel()
			tn = round(tn,3)
			fp = round(fp,3)
			fn = round(fn,3)
			tp = round(tp,3)
			acc = round((tp + tn)/(tp + tn + fp + fn),3)
			recall = round(tp/(tp+fn),3)
			precision = round(tp/(tp+fp),3)
			recall_list.append(recall)
			precision_list.append(precision)
			acc_list.append(acc)
			print(fake_in_str)
			print("Accuracy: ", acc)
			count = count + 1
		except ValueError:
			#print('error:',fake_in_str)
			blank = blank + 1
			pass
		

	avgAcc = np.nanmean(acc_list)
	avgRecall = np.nanmean(recall_list)
	avgPrecise = np.nanmean(precision_list)
	F1Score = 2/((1/avgRecall)+(1/avgPrecise))
	print("Average Accuracy is: " , avgAcc,file=out_file)
	print("Blank data number: ", blank,file=out_file)
	print("Data number: ",count,file=out_file)
	print("Recall: ",avgRecall,file=out_file)
	print("Precise: ",avgPrecise,file=out_file)
	print("F1Score: ",F1Score,file=out_file)
	out_file.close()

		# #copy&paste images with accuracy less than 80%
		# if(acc < 0.8):
		# 	copy2(real_in_str, 'LessThan80/')
		# 	copy2(fake_in_str, 'LessThan80/')
		# 	copy2(realA_in_str, 'LessThan80/')

		# 	#record the number of images
		# 	if product_name in product_dic:
		# 		product_dic[product_name] = product_dic[product_name] + 1
		# 	else:
		# 		product_dic[product_name] = 1

		# 	#print accuracy in the file
		# 	print(fake_in_str, file=out_file)
		# 	print("Accuracy: ",acc, file=out_file)
	# print("++++++++++++++++++++++++++++++++++", file=out_file)
	# print("Summary: ", file=out_file)
	# for x in product_dic:
	# 	print(x, file=out_file)
	# 	print(product_dic[x], file=out_file)
	# 	print("-------------------------------------", file=out_file)
	# print("++++++++++++++++++++++++++++++++++", file=out_file)
	# out_file.close()

""" Generating random shapes images and corresponding binary masks """
def generate_shape():
	for i in range (500):
		image_name = "shape_" + str(i) + ".png"
		#create shape image with color
		shape, _ = random_shapes((256,256), min_shapes=5, max_shapes=10, min_size=20, allow_overlap=True)
		shape_img = Image.fromarray(shape)
		shape_img.save("shape_color/"+image_name,"PNG")

		#create mask
		mask_img = shape_img.convert("L")
		mask_array = np.asarray(mask_img)
		mask_array = np.where(mask_array < 255, 0,255)
		mask_img = Image.fromarray(mask_array.astype(np.uint8))
		mask_img.save("shape_mask/"+image_name,"PNG")

def generate_mask():
	'''Generate fake cloud masks'''
	paths = Path('cloud_few').glob('*.png')
	for path in paths:
		#real_B path
		path_in_str = str(path)
		basename = os.path.basename(path)
		im = Image.open(path_in_str).convert("L")
		imdata = np.asarray(im)
		imdata = np.where(imdata > 150,255,0)
		mask_img = Image.fromarray(imdata.astype(np.uint8))
		mask_img.save("cloud_few_mask/"+basename,"PNG")

def binary_transformation(folder_A):
	'''convert the image into binary masks'''
	paths = Path(folder_A).glob('*fake_B.png')
	try:
		os.mkdir('./'+folder_A+'/binary_fake')
		print("Create results folder binary results folder")
	except OSError as error:
		print(error)
	for path in paths:
		im = Image.open(str(path))
		basename = os.path.basename(path)
		imdata = np.asarray(im)
		imdata = np.where(imdata > 140,255,0)
		mask_img = Image.fromarray(imdata.astype(np.uint8))
		mask_img.save(folder_A+"/binary_fake/"+basename,"PNG")


def search_threshold(folder):
	# paths = Path('test').glob('*fake_B.png')
	#open a txt file for recording
	out_file = open("output.txt", "a")
	print("----------------------------------------folder:",folder,"---------------------------------------",file=out_file)
	threshold = 120
	x_axis = []
	y_yxis = []
	for count in range(70):
		print("start processing threshold: ",threshold)
		acc_list = []				
		processed_image_count = 0 #number of images
		paths = Path(folder).glob('*fake_B.png')

		for path in paths:
			#paths
			basename = os.path.basename(path)
			fake_in_str = str(path)
			real_path = re.sub('\.png$', '', fake_in_str)
			real_path = re.sub('fake_B','',real_path)
			real_in_str = real_path + "real_B.png"

			#read real image
			real = Image.open(real_in_str).convert("L")
			data_real = np.asarray(real)
			r = np.where(data_real > 128,1,0)

			#read fake image
			fake = Image.open(fake_in_str).convert("L")
			data_fake = np.asarray(fake)
			f = np.where(data_fake > threshold,1,0)

			#---------------
			imdata = np.where(data_fake > threshold,255,0)
			mask_img = Image.fromarray(imdata.astype(np.uint8))
			mask_img.save("test_output/"+str(threshold)+"_"+basename,"PNG")
			#---------------

			r = r.flatten()
			f = f.flatten()

			try:
				tn, fp, fn, tp = confusion_matrix(r,f).ravel()
				acc = (tp + tn)/(tp + tn + fp + fn)
				processed_image_count = processed_image_count + 1
				acc_list.append(acc)
			except ValueError:
				print("error")
				pass
		print(acc_list)
		print("Threshold: " ,threshold, " Accuracy: ",mean(acc_list), "Number of images:", processed_image_count, file=out_file)
		x_axis.append(threshold)
		y_yxis.append(mean(acc_list))
		threshold = threshold + 1 #threshold
	out_file.close()
	plt.plot(x_axis,y_yxis)
	plt.xlabel('Threshold')
	plt.ylabel('Accuracy')
	plt.show()
			

#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def caculate_AtoB(folder_A):
	'''Caculate PSNR and SSIM of synthetic images'''
	out_file = open("acc_output.txt", "a")
	print("----------------------------------------folder:",folder_A,"---------------------------------------",file=out_file)
	#Get images paths
	real_paths = Path(folder_A).glob('*real_B.png')
	#open a txt file for recording
	# out_file = open("output.txt", "a")

	for real_path in real_paths:
		#real_B path
		real_in_str = str(real_path)
		#fake_B path
		fake_path = re.sub('\.png$', '', real_in_str)
		fake_path = re.sub('real_B','',fake_path)
		fake_in_str = fake_path + "fake_B.png"


		#product name
		product_name = re.search('S3A_(.*)SEN3',real_in_str).group(1)
		product_name = "S3A_" +product_name+ "SEN3"

		#read fake image
		#fake_in_str = re.sub(folder_A+'/',folder_A+'/binary_fake/',fake_in_str)
		print("fake",fake_in_str)
		print("real",real_in_str)

		fake = Image.open(fake_in_str)
		data_fake = np.asarray(fake)

		#read real image
		real = Image.open(real_in_str)
		data_real = np.asarray(real)

		print("------image: ", fake_in_str,"---------",file=out_file)
		#ssim
		ssim_value = compare_ssim(real, fake)
		print("ssim: ",ssim_value,file=out_file)
		PSNR_value = PSNR(data_real,data_fake)
		print("PSNR: ",PSNR_value,file=out_file)

	out_file.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--folder_A", type=str,help="input folder path")
	parser.add_argument("--folder_B", type=str,help="output folder path")
	parser.add_argument("--search_threshold", action="store_true",help="search threshold")
	parser.add_argument("--binary_transformation", action="store_true",help="bianry transformation")
	parser.add_argument("--caculate_acc", action="store_true",help="caculate accuracy")
	parser.add_argument("--caculate_AtoB", action="store_true",help="caculate ssim and PSNR")
	args = parser.parse_args()

	if (args.binary_transformation):
		#python ImageClassify.py --binary_transformation --folder_A test_results/lambda_80 --folder_B test_results/lambda_80/binary_fake
		binary_transformation(args.folder_A)
	elif (args.search_threshold):
		search_threshold(args.folder_A)
	elif (args.caculate_acc):
		#python ImageClassify.py --caculate_acc --folder_A test_results/BCELoss_reflection_padding
		caculate_accuracy(args.folder_A)
	elif (args.caculate_AtoB):
		#python ImageClassify.py --caculate_AtoB --folder_A test_results/sample
		caculate_AtoB(args.folder_A)

if __name__== "__main__":
	main()
