--[[This code specify the model for MNIST dataset. 
In this file we also secify the Glorot learning parameter]]
require 'nn'
require '../newLayers/BinaryLinear.lua'
require '../newLayers/BinarizedNeurons'

if opt.type=='cuda' then
  require 'cunn'
  require 'cudnn'
end

local BatchNormalization = nn.BatchNormalization

local model = nn.Sequential()
local numHid =2048

model:add(nn.View(-1,784))

model:add(BinaryLinear(784,numHid))
model:add(BatchNormalization(numHid))
model:add(nn.HardTanh())
model:add(BinarizedNeurons())

model:add(BinaryLinear(numHid,numHid))
model:add(BatchNormalization(numHid))
model:add(nn.HardTanh())
model:add(BinarizedNeurons())

model:add(BinaryLinear(numHid,numHid))
model:add(BatchNormalization(numHid))
model:add(nn.HardTanh())
model:add(BinarizedNeurons())

model:add(BinaryLinear(numHid,10))
model:add(nn.BatchNormalization(10))



local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)

local counter = 0
for i, layer in ipairs(model.modules) do
   if layer.__typename == 'BinaryLinear' then
      local weight_size = layer.weight:size(1)*layer.weight:size(2)
      local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      counter = counter+bias_size
    elseif layer.__typename == 'nn.BatchNormalization' then
      local weight_size = layer.weight:size(1)
      local size_w=layer.weight:size()
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      counter = counter+bias_size
  end
end

print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(counter)

return {
   model = model,
   lrs = learningRates
}
