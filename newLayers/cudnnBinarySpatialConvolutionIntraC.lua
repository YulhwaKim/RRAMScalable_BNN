require './cudnnSpatialConvolutionIntraC.lua'

local cudnnBinarySpatialConvolutionIntraC, parent =
    torch.class('cudnnBinarySpatialConvolutionIntraC', 'cudnn.SpatialConvolutionIntraC')
local ffi = require 'ffi'
local errcheck = cudnn.errcheck

local autotunerCache = {}
autotunerCache[1] = {} -- forward
autotunerCache[2] = {} -- backwardFilter
autotunerCache[3] = {} -- backwardData

function cudnnBinarySpatialConvolutionIntraC:__init(nInputPlane, nOutputPlane,
                            kW, kH, dW, dH, padW, padH, groups, nMaxInputPlaneGroup)
    local delayedReset = self.reset
    local delayedMasking = self.weightMasking
    self.weightMasking = function() end
    self.reset = function() end
    parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH)
    self.reset = delayedReset
    self.weightMasking = delayedMasking
    self.padW = padW or 0
    self.padH = padH or 0
    self.groups = groups or 1
    print(self.groups)
    assert(nOutputPlane % self.groups == 0,
           'nOutputPlane should be divisible by nGroups')
    self.nMaxInputPlaneGroup = nMaxInputPlaneGroup or (nInputPlane / self.groups) -- maximum nInputPlane for each group.
    self.weight = torch.Tensor(nOutputPlane, self.nMaxInputPlaneGroup, kH, kW)
    self.weightB = torch.Tensor(nOutputPlane, self.nMaxInputPlaneGroup, kW, kH)
    self.weightOrg = torch.Tensor(nOutputPlane, self.nMaxInputPlaneGroup, kW, kH)
    self.gradWeight = torch.Tensor(nOutputPlane, self.nMaxInputPlaneGroup, kH, kW)
    self.nInputPlaneMoveGroup = torch.Tensor(self.groups)
    self:reset()
    self:weightMasking(kW, kH)
    -- should nil for serialization, the reset will still work
    self.reset = nil
end

function cudnnBinarySpatialConvolutionIntraC:binarized()
  self.weightOrg:copy(self.weight)
  self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)
  self.weightB:round():mul(2):add(-1) -- no zero
  self.weightB:cmul(self.mask) -- with zero (mask)
  return  self.weightB
end

-- if you change the configuration of the module manually, call this
function cudnnBinarySpatialConvolutionIntraC:resetWeightDescriptors()
     -- for compatibility
    self.groups = self.groups or 1
    assert(cudnn.typemap[torch.typename(self.weight)], 'Only Cuda supported duh!')
    assert(cudnn.typemap[torch.typename(self.bias)] or not self.bias, 'Only Cuda supported duh!')

    -- create descriptor for bias
    if self.bias then
        self.biasDesc = cudnn.toDescriptor(self.bias:view(1, self.nOutputPlane,1,1))
    end

    self.weightDesc = cudnn.setFilterDescriptor(
       { dataType = cudnn.typemap[torch.typename(self.weight)],
         filterDimA = desc or
            {self.nOutputPlane/self.groups,
             self.nMaxInputPlaneGroup,
             self.kH, self.kW}
       }
    )
    self.weightDesc_last = cudnn.setFilterDescriptor(
    { dataType = cudnn.typemap[torch.typename(self.weight)],
      filterDimA = desc or 
        {self.nOutputPlane/self.groups,
        self.nInputPlaneMoveGroup[self.groups],
        self.kH, self.kW}
      }
    )
    return self
end

function cudnnBinarySpatialConvolutionIntraC:fastest(mode)
    if mode == nil then mode = true end
    self.fastest_mode = mode
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function cudnnBinarySpatialConvolutionIntraC:setMode(fmode, bdmode, bwmode)
    if fmode ~= nil then
        self.fmode = fmode
    end
    if bdmode ~= nil then
        self.bdmode = bdmode
    end
    if bwmode ~= nil then
        self.bwmode = bwmode
    end
    self.iSize = self.iSize or torch.LongStorage(4)
    self.iSize:fill(0)
    return self
end

function cudnnBinarySpatialConvolutionIntraC:resetMode()
    self.fmode = nil
    self.bdmode = nil
    self.bwmode = nil
    return self
end

function cudnnBinarySpatialConvolutionIntraC:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function cudnnBinarySpatialConvolutionIntraC:createIODescriptors(input)
    parent.createIODescriptors(self,input)
end

local one = torch.FloatTensor({1});
local zero = torch.FloatTensor({0});

local function makeContiguous(self, input, gradOutput)
   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:typeAs(input):resizeAs(input):copy(input)
      input = self._input
   end
   if gradOutput and not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:typeAs(gradOutput):resizeAs(gradOutput):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   return input, gradOutput
end

function cudnnBinarySpatialConvolutionIntraC:updateOutput(input)
    self.weightOrg:copy(self.weight)
    self.weightB = self:binarized()
    self.weight:copy(self.weightB)
    parent.updateOutput(self,input)
    self.weight:copy(self.weightOrg)
    return self.output
end

function cudnnBinarySpatialConvolutionIntraC:updateGradInput(input, gradOutput)
    if not self.gradInput then return end
    self.weight:copy(self.weightB)
    parent.updateGradInput(self, input, gradOutput:contiguous(), scale)
    self.weight:copy(self.weightOrg)
    return self.gradInput
end

function cudnnBinarySpatialConvolutionIntraC:accGradParameters(input, gradOutput, scale)
    parent.accGradParameters(self, input, gradOutput:contiguous(), scale)
end

function cudnnBinarySpatialConvolutionIntraC:clearDesc()
    self.weightDesc = nil
    self.weightDesc_last = nil
    self.biasDesc = nil
    self.convDesc = nil
    self.iDesc = nil
    self.iDesc_last = nil
    self.oDesc = nil
    self.oDescForBias = nil
    self.algType = nil
    self.fwdAlgType = nil
    self.bwdDataAlgType = nil
    self.bwdFilterAlgType = nil
    self.extraBuffer = nil
    self.extraBufferSizeInBytes = nil
    self.scaleT = nil
end

function cudnnBinarySpatialConvolutionIntraC:write(f)
    self:clearDesc()
    local var = {}
    for k,v in pairs(self) do
        var[k] = v
    end
    f:writeObject(var)
end

function cudnnBinarySpatialConvolutionIntraC:clearState()
   self:clearDesc()
   return nn.Module.clearState(self)
end
