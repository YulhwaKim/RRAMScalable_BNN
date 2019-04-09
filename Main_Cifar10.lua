-- th Main_Cifar10.lua -network spCifar10_Model -nGPU 2 -arraySize 256

require 'torch'
require 'xlua'
require 'optim'
require 'gnuplot'
require 'pl'
require 'trepl'
require 'adaMax_binary_clip_shift'
require 'nn'
require 'SqrHingeEmbeddingCriterion'
require 'optim'
----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'Model.lua',            'Model file - must return valid network.')
cmd:option('-LR',                 2^-6,                   'learning rate')
cmd:option('-LRDecay',            0.5,                      'learning rate decay')
cmd:option('-weightDecay',        0.0,                    'L2 penalty on the weights')
cmd:option('-momentum',           0.0,                    'momentum')
cmd:option('-batchSize',          200,                    'batch size')
cmd:option('-optimization',       'adaMax',                 'optimization method')
cmd:option('-epoch',              500,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-normalization',      'simple',               'simple - whole sample, channel - by image channel, image - mean and std images')
cmd:option('-format',             'rgb',                  'rgb or yuv')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')
-- for splitting
cmd:option('-arraySize',          512,                    'The size of target rram array')
cmd:option('-optnet',             false,                  'Optimize the memory of the network')
cmd:text('===>Misc')

torch.manualSeed(432)
opt = cmd:parse(arg or {})
local netFilename = paths.concat('/home/yulhwa.kim', opt.network .. '_' .. opt.arraySize) -- netFile save path

local resultFolder = './Results'
opt.save = paths.concat(resultFolder, opt.network, opt.arraySize, opt.save)

opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')


os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.augment then
    require 'image'
end
----------------------------------------------------------------------
-- Model + Loss:
local modelAll = require(opt.network)
model=modelAll.model
GLRvec=modelAll.lrs
clipV=modelAll.clipV

local loss = SqrtHingeEmbeddingCriterion(1)


local data = require 'Data'
local classes = data.Classes

----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local AllowVarBatch = not opt.constBatchSize

----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local logFilename = paths.concat(opt.save,'AccRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------

local TensorType = 'torch.FloatTensor'
if paths.filep(opt.load) then
    model = torch.load(opt.load)
    print('==>Loaded model from: ' .. opt.load)
    print(model)
end
if opt.type =='cuda' then
    require 'cutorch'
    cutorch.setDevice(opt.devid)
    cutorch.setHeapTracking(true)
    model:cuda()
    GLRvec=GLRvec:cuda()
    loss = loss:cuda()
    TensorType = 'torch.CudaTensor'

    -- turn on this if you want to set fastest mode on cudnn
    -- cudnn.fastest = true
    -- cudnn.benchmark = true
end

-- optimizet the network memory
if opt.optnet then
  local optnet = require 'optnet'
  local imsize = 32
  local sampleInput = torch.zeros(4,3,imsize,imsize):type('torch.CudaTensor')
  optnet.optimizeMemory(model, sampleInput, {inplace=false, mode = 'training'})
end


---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end


-- Optimization configuration
local Weights,Gradients = model:getParameters()

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)



------------------Optimization Configuration--------------------------
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay,
    GLRvec=GLRvec
}
----------------------------------------------------------------------

-- return image itself 
local function SampleTestImages(images,labels)
  return images,labels
end

-- 4 pixels are padded on each side
-- 32x32 crop is randomly sampled from the padded image
-- randomly conduct horizontal flop
local function SampleTrainImages(images,labels)
    if not opt.augment then
        return images,labels
    else
      local sampled_imgs = images:clone()
      -- local sampled_imgs = torch.Tensor():typeAs(images):resizeAs(images)
      local bs = images:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,images:size(1) do
          local temp_image = torch.zeros(3,40,40)
          temp_image[{{}, {5,36}, {5,36}}]:copy(images[i])


          local startx = math.random(8)
          local starty = math.random(8)
          local img = temp_image:narrow(2,starty,32):narrow(3,startx,32)
          if flip_mask[i] then
              img = image.hflip(img)
          end
          sampled_imgs[i]:copy(img)
      end
      return sampled_imgs,labels
    end
end


------------------------------
local function Forward(Data, train)

  local MiniBatch
  if train then
    MiniBatch = DataProvider.Container{
      Name = 'GPU_Batch',
      MaxNumItems = opt.batchSize,
      Source = Data,
      ExtractFunction = SampleTrainImages,
      TensorType = TensorType
    }
  else 
    MiniBatch = DataProvider.Container{
      Name = 'GPU_Batch',
      MaxNumItems = opt.batchSize,
      Source = Data,
      ExtractFunction = SampleTestImages,
      TensorType = TensorType
    }
  end

  local yt = MiniBatch.Labels
  local x = MiniBatch.Data
  local SizeData = Data:size()
  if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end

  local NumSamples = 0
  local NumBatches = 0
  local lossVal = 0

  while NumSamples < SizeData do
    MiniBatch:getNextBatch()
    local y, currLoss
    NumSamples = NumSamples + x:size(1)
    NumBatches = NumBatches + 1
    if opt.nGPU > 1 then
      model:syncParameters()
    end
    y = model:forward(x)
    one_hot_yt=torch.zeros(yt:size(1),10)
    one_hot_yt:scatter(2, yt:long():view(-1,1), 1)
    one_hot_yt=one_hot_yt:mul(2):float():add(-1)
    if opt.type == 'cuda' then
      one_hot_yt=one_hot_yt:cuda()
    end

    currLoss = loss:forward(y,one_hot_yt)
    if train then
      function feval()
        model:zeroGradParameters()
        local dE_dy = loss:backward(y, one_hot_yt)
        model:backward(x, dE_dy)
        return currLoss, Gradients
      end

       adaMax_binary_clip_shift(feval, Weights, optimState)
    end

    lossVal = currLoss + lossVal

    if type(y) == 'table' then --table results - always take first prediction
      y = y[1]
    end

    confusion:batchAdd(y,one_hot_yt)
    xlua.progress(NumSamples, SizeData)
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
  end
  return(lossVal/math.ceil(SizeData/opt.batchSize))
end

------------------------------
local function Train(Data)
  model:training()
  return Forward(Data, true)
end

local function Test(Data)
  model:evaluate()
  return Forward(Data, false)
end
------------------------------

local epoch = 1
print '\n==> Starting Training\n'

-- reset the best acc
local best_acc = 0
local best_epoch = 0


while epoch ~= opt.epoch do
    data.TrainData:shuffleItems()
    print('Epoch ' .. epoch)
    --Train
    confusion:zero()
    local LossTrain = Train(data.TrainData)
    confusion:updateValids()
    local AccTrain = confusion.totalValid
    print('Training Acc = ' .. AccTrain)
    print('Training Loss = ' .. LossTrain)

    --validation
    confusion:zero()
    local LossValid = Test(data.ValidData)
    confusion:updateValids()
    local AccValid = confusion.totalValid
    print('Valid Acc = ' .. AccValid)
    print('Valid Loss = ' .. LossValid)

    --Test
    confusion:zero()
    local LossTest = Test(data.TestData)
    confusion:updateValids()
    local AccTest = confusion.totalValid
    print('Test Acc = ' .. AccTest)
    print('Test Loss = ' .. LossTest)

    -- when the test acc is the best, save the model
    if best_acc < AccValid then
      best_acc = AccValid
      best_epoch = epoch
      if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(netFilename, model:clearState():get(1))
      else
        torch.save(netFilename, model:clearState())
      end
    end
    print('best Acc = ' .. best_acc)
    print('best epoch = ' .. best_epoch)

    Log:add{['Training Acc']= AccTrain, ['Valid Acc'] = AccValid, ['Test Acc'] = AccTest}
    if epoch%50==0  then
      optimState.learningRate=optimState.learningRate*opt.LRDecay
    else
      optimState.learningRate=optimState.learningRate
    end
    print('-------------------LR-------------------')
    print(optimState.learningRate)
    epoch = epoch + 1
end
