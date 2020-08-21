# DissertationProject
#### These two scripts are used for the dissertation project.
-  slstr-preprocess.py is used for prepare a dataset.
-  ImageClassify.py is used for analysing the results.


#### slstr-preprocess.py
- Extract the Bayesian Mask and Natural Color Images from SLSTR products
  - folder_A is the location of SLSTR packages, they should have "S3A*" at the begining of the name.
  - folder_B is the location you want to store Bayesian Masks. There is also a function for extract Summary Masks in the code. By default, we are extract Bayesian Masks.
  - folder_C is the location you want to store the natural color satellite images. 
  

        python3 slstr_preprocess.py --read_product --folder_A data --folder_B bayes_mask --folder_C natural_image
        
- Resize the masks into the same size of satellite images
  - folder_A is the original masks.
  - folder_B is the location you want to store the resized masks.

        python3 slstr_preprocess.py --resize_mask --folder_A bayes_mask --folder_B bayes_mask/resized_mask
        
- Crop the images into small pieces
  - folder_A is the source folder.
  - folder_B is the location you want to store the cropped images.
  - window_size is the side of the square. e.g. 256*256
  
        python3 slstr_preprocess.py --crop_image --folder_A bayes_mask/resized_mask --folder_B bayes_mask/cropped_mask --window_size 256
        python3 slstr_preprocess.py --crop_image --folder_A natural_image --folder_B natural_image/cropped_natural --window_size 256
        
- Remove pure white and black masks. And the corresponding natural color images.
   - folder_A is the masks.
   - folder_B is the natural color images.
 
        python3 slstr_preprocess.py --remove_bw --folder_A bayes_mask/cropped_mask --folder_B natural_image/cropped_natural
        
- If you are using a Mac, you might need to remove all '.DC_Store' files before running the combination code.

        find . -name ".DS_Store" -delete
  
- Then you can run the code from pix2pix to combine your images

        python3 datasets/combine_A_and_B.py --fold_A datasets/aligned/A --fold_B datasets/aligned/B --fold_AB datasets/AtoB_lambda_AB

#### ImageClassify.py

- search threshold. This command can find the best value for binnary transformation  from 120-190. This could take a while. From my experiment, 180 usually achieves the best results.
  - folder_A is the results of pix2pix


        python3 ImageClassify.py --search_threshold --folder_A test_results/BCELoss

- binary transformation. This is default using 180 as threshold
  - folder_A is the results of pix2pix. This will save the bianary results inside the folder_A. e.g. test_results/BCELoss/binary_fake
        
        python3 ImageClassify.py --binary_transformation --folder_A test_results/BCELoss

- caculate accuracy, recall, precision and F1 Score. The output will be out print into a txt called acc_output.txt. 
  - folder_A is the results of pix2pix.
    
        python3 ImageClassify.py --caculate_acc --folder_A test_results/BCELoss
        
- caculate PSNR and SSIM. This is used for the direction "mask to image"
  - folder_A is the results of pix2pix.
    
        python3 ImageClassify.py --caculate_AtoB --folder_A test_results/sample

