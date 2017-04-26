--[[
This script creates a training set and a testing set from an input dataset. It does the followings:
- Reads each raw image from the input dataset and associates it with the correct label
- Splits these labeled images into training and testing sets
- Saves the data information on each set into a training or a testing tensor so that
Torch can load it later for training and testing purposes
- The training and a testing tensors are also saved on your local machine under t7
format
]]

require 'nn';
require 'image';
require 'torchx';
cutorch = require 'cutorch'

--DIRECTORIES OF THE DATASET
early_good="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Good"
early_avg="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Average"
early_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Early/Poor"
mid_good = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Good"
mid_avg = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Average"
mid_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Mid/Poor"
late_good = "/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Good"
late_avg="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Average"
late_poor="/home/superuser/project/Preprocessed_photos/Volumes/External/Picture/Late/Poor"

ivch = 3                --#channel
desImaX = 32            --size of the length dimension 
desImaY = 32            --size of the width dimension 

trainData = {
                data = torch.Tensor(train_size, ivch,desImaX,desImaY),      
                labels = torch.Tensor(train_size),                          
            }
testData =  {
                data = torch.Tensor(test_size, ivch,desImaX,desImaY),       
                labels = torch.Tensor(test_size),               
            }

--create a list of images in a directory 
--i.e. add filenames in a folder to a table for readability  
function dir_to_list(dir)
	label = {}
	files = {}
	for file in paths.files(dir) do
		if (file:find('jpg') or file:find('JPG') or file:find('png') or file:find('PNG') or file:find('jpeg') or file:find('JPEG')) then
		table.insert(files,paths.concat(dir,file))
		end
	end
	return files
end

-- return size of the training set and size of the testing set at each directory
function split_dataset (files)
        --2/3 train, 1/3 test
        total_images = #files
        train_size = math.modf((total_images/3)*2)
        test_size = total_images - train_size
        return  train_size, test_size
end

-- return new index of the training set and new index of the testing set after saving the images from the previous directory 
function get_index (old_index_tr,old_index_te,train_size,test_size)
        new_index_tr = old_index_tr + train_size*3
        new_index_te = old_index_te + test_size*3
        return new_index_tr, new_index_te
end

--create training and testing sets:
--read the input images from the data directories 
--split them into training and testing sets 
--artificially increase the amount of the input images (if applicable)
--associate the images with the correct labels 
--save these information into the trainData and the testData tensors 
function image_to_t7(my_files, index_train,index_test)
	files = {}
	files = dir_to_list(my_files)
	images = {}

    --Split the data into training and testing set 
	train_size, test_size = split_dataset(files)

	for i,file in ipairs(files) do
        --create a training dataset
		if(i<=train_size) then	
			my_image = image.load(file)
			clone_bl = image.load(file)	
			clone_br = image.load(file)
			clone_ct = image.load(file)
            
            -- To create squared-original training set, uncomment this part 
            --[[
            nchan, height, width = my_image:size(1), my_image:size(2), my_image:size(3)
            -- find min dimension 
            if height<width then
                dim_min = height
             else
                dim_min = width
            end
			my_image = image.crop(my_image,"c",dim_min,dim_min)
            my_image = image.scale(my_image,32,32,'bilinear')
            trainData.data[index_train] = my_image
            --]]

            -- To create squared-original training set, comment out this part 
            -- Create ordinary-original training set
			my_image = image.scale(my_image,32,32,'bilinear')    -- resize the input images into 32 x 32 
			trainData.data[index_train] = my_image               -- save the input data to trainData tensor 

            --setting labels 
            --label = 1 if Good, 2 if Average and 3 if Poor
			if file:find('Good') then
				trainData.labels[index_train] = 1                -- save the input data to trainData tensor 
			else if file:find('Average') then
				trainData.labels[index_train] = 2                -- save the input data to trainData tensor 
			else
				trainData.labels[index_train] = 3                -- save the input data to trainData tensor 
			end
			end		
			true_label = trainData.labels[index_train]		-- true label for this batch of clones
			index_train = index_train + 1

            --Data augmentation 
			--CROP THE INPUT IMAGAE AT THE BOTTOM LEFT --
			clone_bl = image.crop(clone_bl,"bl",400, 400)     -- Cropped size: 400 x 400
			my_image = image.scale(clone_bl,32,32,'bilinear') -- resize the input images into 32 x 32 
			trainData.data[index_train] = my_image           -- save the input data to trainData tensor 
			trainData.labels[index_train] = true_label       -- save the input data to trainData tensor 
			index_train = index_train + 1

			--CROP THE INPUT IMAGAE AT THE BOTTOM RIGHT --
			clone_br = image.crop(clone_br,"br",400, 400)     -- Cropped size: 400 x 400
			my_image = image.scale(clone_br,32,32,'bilinear') -- resize the input images into 32 x 32 
			trainData.data[index_train] = my_image           -- save the input data to trainData tensor
            trainData.labels[index_train] = true_label       -- save the input data to trainData tensor
            index_train = index_train + 1

            --CROP THE INPUT IMAGAE AT THE CENTER --
            clone_c = image.crop(clone_ct,"c",400, 400)      -- Cropped size: 400 x 400
			my_image = image.scale(clone_c,32,32,'bilinear') -- resize the input images into 32 x 32 
			trainData.data[index_train] = my_image           -- save the input data to trainData tensor
            trainData.labels[index_train] = true_label       -- save the input data to trainData tensor
            index_train = index_train + 1
            print ('index_train: ', index_train)             -- for debugging 

        --create a testing dataset
		else if(i>train_size)then
            my_image = image.load(file)
    		clone_image = image.load(file)
            
            -- To create squared-original testing set, uncomment this part 
            --[[
            nchan, height, width = my_image:size(1), my_image:size(2), my_image:size(3)
            -- find min dimension 
            if height<width then
                dim_min = height
             else
                dim_min = width
            end
            my_image = image.crop(my_image,"c",dim_min,dim_min)
            my_image = image.scale(my_image,32,32,'bilinear')
            testData.data[index_test] = my_image
            --]]   

            -- To create squared-original testing set, comment out this part 
            -- Create ordinary-original testing set
            my_image = image.scale(my_image,32,32,'bilinear')    -- resize the input images into 32 x 32 
            testData.data[index_test] = my_image                 -- save the input data to testData tensor 

            --setting labels 
            --label = 1 if Good, 2 if Average and 3 if Poor
            if file:find('Good') then
        		testData.labels[index_test] = 1                  -- save the input data to testData tensor 
       		else if file:find('Average') then              
              	testData.labels[index_test] = 2                  -- save the input data to testData tensor 
        	else
        		testData.labels[index_test] = 3                  -- save the input data to testData tensor 
        	end
        	end
    		true_label = testData.labels[index_test]             -- true label for this batch of clones
            index_test = index_test + 1

            --Data augmentation 
            --CROP THE INPUT IMAGAE AT THE BOTTOM LEFT --
    		clone_bl = image.crop(clone_image,"c",400, 400)      -- Cropped size: 400 x 400
            my_image = image.scale(clone_bl,32,32,'bilinear')    -- resize the input images into 32 x 32 
            testData.data[index_test] = my_image                 -- save the input data to testData tensor 
            testData.labels[index_test] = true_label             -- save the input data to testData tensor 
            index_test = index_test + 1

    		--CROP THE INPUT IMAGAE AT THE BOTTOM RIGHT --
            clone_br = image.crop(clone_image,"bl",400, 400)     -- Cropped size: 400 x 400
            my_image = image.scale(clone_br,32,32,'bilinear')    -- resize the input images into 32 x 32 
            testData.data[index_test] = my_image                 -- save the input data to testData tensor 
            testData.labels[index_test] = true_label             -- save the input data to testData tensor 
            index_test = index_test + 1

			--CROP THE INPUT IMAGAE AT THE CENTER --
            clone_c = image.crop(clone_image,"br",400, 400)      -- Cropped size: 400 x 400
            my_image = image.scale(clone_c,32,32,'bilinear')     -- resize the input images into 32 x 32 
            testData.data[index_test] = my_image                 -- save the input data to testData tensor 
            testData.labels[index_test] = true_label             -- save the input data to testData tensor 
            index_test = index_test + 1
            print ('index_test: ',index_test)                    -- for debugging 
		end
		end		
	end
end 

--SEASON EARLY GOOD
files_eg = {}
files_eg = dir_to_list(early_good)		
--read the input images starting with training index = 1 and testing index = 1
image_to_t7(early_good,1,1)
--get the size of the training and the testing set of season "early good"
train_size_eg, test_size_eg = split_dataset(files_eg)

--SEASON EARLY AVG
files_ea = {}
files_ea = dir_to_list(early_avg)
--get the next staring training and testing indices using the size of the training and the testing set of season "early good"
index_train_ea,index_test_ea = get_index(1,1,train_size_eg,test_size_eg)
--read the input images starting with the new training and testing indices
image_to_t7(early_avg,index_train_ea,index_test_ea)
--get the size of the training and the testing set of season "early average"
train_size_ea, test_size_ea = split_dataset(files_ea)

--SEASON EARLY POOR
files_ep = {}
files_ep = dir_to_list(early_poor)
--get the next staring training and testing indices using the size of the training and the testing set of season "early average"
index_train_ep,index_test_ep = get_index(index_train_ea,index_test_ea,train_size_ea, test_size_ea)
--read the input images starting with the new training and testing indices
image_to_t7(early_poor,index_train_ep,index_test_ep)       
--get the size of the training and the testing set of season "early poor"
train_size_ep, test_size_ep = split_dataset(files_ep)

--SEASON MID GOOD
files_mg = {}
files_mg = dir_to_list(mid_good)
--get the next staring training and testing indices using the size of the training and the testing set of season "early poor"
index_train_mg,index_test_mg = get_index(index_train_ep,index_test_ep,train_size_ep, test_size_ep)          
--read the input images starting with the new training and testing indices
image_to_t7(mid_good,index_train_mg,index_test_mg)
--get the size of the training and the testing set of season "mid good"
train_size_mg, test_size_mg = split_dataset(files_mg)

--SEASON MID AVG
files_ma = {}
files_ma = dir_to_list(mid_avg)
--get the next staring training and testing indices using the size of the training and the testing set of season "mid good"
index_train_ma,index_test_ma = get_index(index_train_mg,index_test_mg,train_size_mg, test_size_mg)
--read the input images starting with the new training and testing indices
image_to_t7(mid_avg,index_train_ma,index_test_ma)
--get the size of the training and the testing set of season "mid average"
train_size_ma, test_size_ma = split_dataset(files_ma)

--SEASON MID POOR
files_mp = {}
files_mp = dir_to_list(mid_poor)
--get the next staring training and testing indices using the size of the training and the testing set of season "mid average"
index_train_mp,index_test_mp = get_index(index_train_ma,index_test_ma,train_size_ma, test_size_ma)          
--read the input images starting with the new training and testing indices
image_to_t7(mid_poor, index_train_mp,index_test_mp)
--get the size of the training and the testing set of season "mid poor"
train_size_mp, test_size_mp = split_dataset(files_mp)

--LATE GOOD
files_lg = {}
files_lg = dir_to_list(late_good)
--get the next staring training and testing indices using the size of the training and the testing set of season "mid poor"
index_train_lg,index_test_lg = get_index(index_train_ea,index_test_ea,train_size_ea, test_size_ea)
--read the input images starting with the new training and testing indices
image_to_t7(late_good, index_train_lg,index_test_lg)
--get the size of the training and the testing set of season "late good"
train_size_lg, test_size_lg = split_dataset(files_lg)

--LATE AVG
files_la = {}
files_la = dir_to_list(late_avg)
--get the next staring training and testing indices using the size of the training and the testing set of season "late good"
index_train_la,index_test_la= get_index(index_train_lg,index_test_lg,train_size_lg, test_size_lg)
--read the input images starting with the new training and testing indices
image_to_t7(late_avg,index_train_la,index_test_la)
--get the size of the training and the testing set of season "late good"
train_size_la, test_size_la = split_dataset(files_la)

--LATE POOR
files_lp = {}
files_lp = dir_to_list(late_poor)
--get the next staring training and testing indices using the size of the training and the testing set of season "late average"
index_train_lp,index_test_lp= get_index(index_train_la,index_test_la,train_size_la, test_size_la)
--read the input images starting with the new training and testing indices
image_to_t7(late_poor,index_train_lp,index_test_lp)         
--get the size of the training and the testing set of season "late average"
train_size_lp, test_size_lp = split_dataset(files_lp)

--save data into t7 format 
torch.save("/home/superuser/project/sugarcane-train.t7", trainData)
torch.save("/home/superuser/project/sugarcane-test.t7", testData)