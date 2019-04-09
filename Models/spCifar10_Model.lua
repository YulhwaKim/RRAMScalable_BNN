--[[This code specify the model for CIFAR 10 dataset. This model splits input for mapping on RRAM-based accelerator.
In this file we also secify the Glorot learning parameter and the which of the learnable parameter we clip ]]
require 'nn'
require '../newLayers/BinaryLinear.lua'
require '../newLayers/BinarizedNeurons'
require '../newLayers/SplitBinaryLinear'
require '../newLayers/MergeLinear.lua'
require '../newLayers/MergeConv.lua'
require '../newLayers/SplitBatchNormalization.lua'
require '../newLayers/SplitSpatialBatchNormalization.lua'
local SpatialConvolution
local SpatialMaxPooling
if opt.type =='cuda' then
  require 'cunn'
  require 'cudnn'
  require '../newLayers/cudnnBinarySpatialConvolutionIntraC.lua'
  SpatialConvolution = cudnnBinarySpatialConvolutionIntraC
  SpatialMaxPooling = cudnn.SpatialMaxPooling
end

local  BatchNormalization = nn.BatchNormalization
local  SpatialBatchNormalization = nn.SpatialBatchNormalization


local model = nn.Sequential()

local function convBlock(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  model:add(SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  model:add(SpatialBatchNormalization(nOutputPlane))
  model:add(nn.HardTanh(-1,1,true))
  model:add(BinarizedNeurons())
end

local function splitBinaryLinear(numIn, numOut)
  local groups = numIn / opt.arraySize
  local groupsOut = numOut * groups
  model:add(SplitBinaryLinear(numIn, numOut, groups))
  model:add(SplitBatchNormalization(groupsOut, groups))
  model:add(nn.HardTanh(-1, 1, true))
  model:add(BinarizedNeurons())
  model:add(MergeLinear(groups))
  model:add(BinarizedNeurons())
end

local function splitConvolution(nInputPlane, nOutputPlane, kW, 
                                kH, dW, dH, padW, padH)
  local input_size = kW * kH * nInputPlane
  local groups = math.ceil(input_size / opt.arraySize)
  while input_size % groups ~= 0 do
    groups = groups + 1
  end
  local inputGroup = input_size / groups
  local nOutputPlaneGroups = nOutputPlane * groups
  local nMaxInputPlaneGroup = math.ceil(nInputPlane / groups)
  if nInputPlane % groups ~= 0 then
    nMaxInputPlaneGroup = nMaxInputPlaneGroup + 1
  end
  model:add(SpatialConvolution(nInputPlane, nOutputPlaneGroups, kW, kH,
                                      dW, dH, padW, padH, groups, nMaxInputPlaneGroup))
  model:add(SplitSpatialBatchNormalization(nOutputPlaneGroups, groups))
  model:add(nn.HardTanh(-1, 1, true))
  model:add(BinarizedNeurons())
  model:add(MergeConv(groups))
  model:add(BinarizedNeurons())
end

local function splitConvolutionMaxpooling(nInputPlane, nOutputPlane, kW, 
                                kH, dW, dH, padW, padH)
  local input_size = kW * kH * nInputPlane
  local groups = math.ceil(input_size / opt.arraySize)
  while input_size % groups ~= 0 do
    groups = groups + 1
  end
  local inputGroup = input_size / groups
  local nOutputPlaneGroups = nOutputPlane * groups
  local nMaxInputPlaneGroup = math.ceil(nInputPlane / groups)
  if nInputPlane % groups ~= 0 then
    nMaxInputPlaneGroup = nMaxInputPlaneGroup + 1
  end
  model:add(SpatialConvolution(nInputPlane, nOutputPlaneGroups, kW, kH,
                                      dW, dH, padW, padH, groups, nMaxInputPlaneGroup))
  model:add(SpatialMaxPooling(2,2))
  model:add(SplitSpatialBatchNormalization(nOutputPlaneGroups, groups))
  model:add(nn.HardTanh(-1, 1, true))
  model:add(BinarizedNeurons())
  model:add(MergeConv(groups))
  model:add(BinarizedNeurons())
end

-- Convolution Layers
convBlock(3, 128, 3, 3 ,1,1,1,1)
splitConvolutionMaxpooling(128, 128, 3, 3,1,1,1,1)

splitConvolution(128, 256, 3, 3 ,1,1,1,1)
splitConvolutionMaxpooling(256, 256, 3, 3 ,1,1,1,1)

splitConvolution(256, 512, 3, 3,1,1,1,1)
splitConvolutionMaxpooling(512, 512, 3, 3,1,1,1,1)

-- FC Layers
numHid=1024

model:add(nn.View(512*4*4))
splitBinaryLinear(512*4*4, numHid)
splitBinaryLinear(numHid, numHid)

-- classification layer
model:add(BinaryLinear(numHid,10))
model:add(nn.BatchNormalization(10))



local dE, param = model:getParameters()
local weight_size = dE:size(1)
local learningRates = torch.Tensor(weight_size):fill(0)
local counter = 0

function lr_update(layer)
  if layer.__typename == 'BinaryLinear' then
    local weight_size = layer.weight:size(1)*layer.weight:size(2)
    local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]))
    GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
    learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
    counter = counter+weight_size
    local bias_size = layer.bias:size(1)
    learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
    counter = counter+bias_size
  elseif layer.__typename == 'SplitBinaryLinear' then
    local weight_size = layer.weight:size(1)*layer.weight:size(2)
    local size_w=layer.weight:size();   GLR=1/torch.sqrt(1.5/(size_w[1]+size_w[2]/layer.groups))
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
  elseif layer.__typename == 'SplitBatchNormalization' then
    local weight_size = layer.weight:size(1)
    local size_w=layer.weight:size()
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
  elseif layer.__typename == 'SplitSpatialBatchNormalization' then
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
    GLR=1/torch.sqrt(1.5/(size_w[1]/layer.groups*filter_size+size_w[2]*filter_size))
    GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
    learningRates[{{counter+1, counter+weight_size}}]:fill(GLR)
    counter = counter+weight_size
    local bias_size = layer.bias:size(1)
    learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
    counter = counter+bias_size
  elseif layer.__typename == 'cudnnBinarySpatialConvolutionIntraC' then
    local size_w=layer.weight:size();
    local weight_size = size_w[1]*size_w[2]*size_w[3]*size_w[4]
    local filter_size=size_w[3]*size_w[4]
    GLR=1/torch.sqrt(1.5/(size_w[1]/layer.groups*filter_size+size_w[2]*filter_size))
    GLR=(math.pow(2,torch.round(math.log(GLR)/(math.log(2)))))
    learningRates[{{counter+1, counter+weight_size}}]:fill(GLR):cmul(layer.mask)
    counter = counter+weight_size
    local bias_size = layer.bias:size(1)
    learningRates[{{counter+1, counter+bias_size}}]:fill(GLR)
    counter = counter+bias_size
  end
end

model:apply(lr_update)

print(learningRates:eq(0):sum())
print(learningRates:ne(0):sum())
print(counter)

return {
  model = model,
  lrs = learningRates
}
