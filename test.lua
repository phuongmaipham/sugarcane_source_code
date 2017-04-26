--[[
This script first loads a testing set created by read_images.lua and a trained network created by train_model.lua. 
It then applies the trained network to the testing set and returns the calculated results.
]]
require 'nn';
require 'image';
require 'torchx';
require 'cunn';

-- get dataset
testset = torch.load('/home/superuser/project/sugarcane-test.t7')
--get model 
net = torch.load("/home/superuser/project/model.t7")
--the names of the correspoding output classes 
--classes = {'good', 'medium', 'bad'}
classes = {'good', 'medium'}
--class_performance = {0, 0, 0}
class_performance = {0, 0}
--function to get the testset tensor size 
function testset:size()
        return self.data:size(1)
end
-- convert from Byte tensor to float tensor to use the CPU 
testset.data = testset.data:float()

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