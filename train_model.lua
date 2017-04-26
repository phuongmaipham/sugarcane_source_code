--[[
This script loads the training and testing sets created by read_images.lua, 
train a network on the training set and evaluate this trained network against the testing set. 
The network configuration can be modified from this file. 
The trained network is then saved on your local machine under t7 format.
]]
require 'nn';
require 'image';
require 'torchx';
require 'cunn';
require 'cudnn';

-- get dataset
trainset = torch.load('/home/superuser/project/sugarcane-train.t7')
testset = torch.load('/home/superuser/project/sugarcane-test.t7')
--the names of the correspoding output classes 
--classes = {'good', 'medium', 'bad'}
classes = {'good', 'medium'}
--class_performance = {0, 0, 0}
class_performance = {0, 0}
--function to get the training set tensor size 
function trainset:size() 
	return self.data:size(1) 
end
--function to get the testing set tensor size 
function testset:size() 
	return self.data:size(1) 
end
--To prepare the dataset to be used with StochasticGradient, a couple of things have to be done according to itS documentation: 
--The dataset has to have a size AND The dataset has to have a [i] index operator, so that dataset[i] returns the ith sample.
setmetatable(trainset, 
	{__index = function(t, i) 
					return {
						t.data[i],
						t.labels[i],
					} 
				end}
);
--set the type of the trainning set to be double 
trainset.data = trainset.data:double()

-- The convolutional neural network configuration is defined here 
net = nn.Sequential()
net:add(nn.SpatialConvolutionMM(3,16,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.SpatialConvolutionMM(16,20,5,5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(20*5*5))
net:add(nn.Linear(20*5*5,120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84,2))
net = net:cuda()

--Process the training set using the GPU 
trainset.data = trainset.data:cuda()

criterion = nn.CrossEntropyCriterion()
criterion = criterion:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.005
-- just do 15 epochs of training.
trainer.maxIteration =15	
trainer:train(trainset)

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
testset.data = testset.data:cuda()

--Print the model's details prediction for each image 
correct = 0
for i=1,testset:size() do
	local groundtruth = testset.labels[i]
	local prediction = net:forward(testset.data[i])
	print('Image number: ', i)
	for i=1,prediction:size(1) do
		print('Lables ', classes[i], prediction[i])
	end
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
		correct = correct + 1
	end
end
--Print the overall percent correct 
print(correct, 100*correct/testset:size().. ' % ')

--Print the percent correct in each class 
for i=1,testset:size() do
	local groundtruth = testset.labels[i]
	local prediction = net:forward(testset.data[i])
	local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
	if groundtruth == indices[1] then
		class_performance[groundtruth] = class_performance[groundtruth] + 1
	end
end
for i=1,#classes do
	print(classes[i], 100*class_performance[i]/testset:size() .. ' %')
end
--The model is saved here 
net = net:float()		--convert to Float tensor in order to train on CPU mode 
torch.save("/home/superuser/project/model.t7",net:clearState())
