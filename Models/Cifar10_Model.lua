--[[This code specify the model for CIFAR 10 dataset. 
In this file we also secify the Glorot learning parameter ]]
require 'nn'
require '../newLayers/BinaryLinear.lua'
require '../newLayers/BinarizedNeurons.lua'

local SpatialConvolution
local SpatialMaxPooling
if opt.type =='cuda' then
  require 'cunn'
  require 'cudnn'
  require '../newLayers/cudnnBinarySpatialConvolution.lua'
  SpatialConvolution = cudnnBinarySpatialConvolution
  SpatialMaxPooling = cudnn.SpatialMaxPooling
end

local BatchNormalization = nn.BatchNormalization
local SpatialBatchNormalization = nn.SpatialBatchNormalization

numHid=1024;
local model = nn.Sequential()

local function convBlock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  model:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  model:add(SpatialBatchNormalization(nOutputPlane))
  model:add(nn.HardTanh(-1,1,true))
  model:add(BinarizedNeurons())
end

local function convMaxBlock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  model:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  model:add(SpatialMaxPooling(2, 2))
  model:add(SpatialBatchNormalization(nOutputPlane))
  model:add(nn.HardTanh(-1,1,true))
  model:add(BinarizedNeurons())
end

local function FCBlock(numIn, numOut)
  model:add(BinaryLinear(numIn,numOut))
  model:add(BatchNormalization(numOut))
  model:add(nn.HardTanh(-1, 1, true))
  model:add(BinarizedNeurons())
end


-- Convolution Layers
convBlock(3, 128, 3, 3 ,1,1,1,1)
convMaxBlock(128, 128, 3, 3,1,1,1,1)

convBlock(128, 256, 3, 3 ,1,1,1,1)
convMaxBlock(256, 256, 3, 3 ,1,1,1,1)

convBlock(256, 512, 3, 3,1,1,1,1)
convMaxBlock(512, 512, 3, 3,1,1,1,1)

-- FC Layers
model:add(nn.View(512*4*4))
FCBlock(512*4*4, numHid)
FCBlock(numHid, numHid)

-- classification layer
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
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
      counter = counter+bias_size
    elseif layer.__typename == 'nn.BatchNormalization' then
      local weight_size = layer.weight:size(1)
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      counter = counter+bias_size
    elseif layer.__typename == 'nn.SpatialBatchNormalization' then
      local weight_size = layer.weight:size(1)
      local size_w=layer.weight:size()
      learningRates[{{counter+1, counter+weight_size}}]:fill(1)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(1)
      counter = counter+bias_size
    elseif layer.__typename == 'cudnnBinarySpatialConvolution' then
      local size_w=layer.weight:size();
      local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]
      local filter_size=size_w[3]*size_w[4]
      GLR=1/torch.sqrt(1.5/(size_w[1]*filter_size+size_w[2]*filter_size))
      GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
      learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
      counter = counter+weight_size
      local bias_size = layer.bias:size(1)
      learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
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
