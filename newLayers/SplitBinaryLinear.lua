require './SplitLinear'

local SplitBinaryLinear, parent = torch.class('SplitBinaryLinear', 'nn.SplitLinear')

function SplitBinaryLinear:__init(inputSize, outputSize, groups)
   local delayedReset = self.reset
   self.reset = function() end
   parent.__init(self, inputSize, outputSize)
   self.reset = delayedReset

   self.weight = torch.Tensor(outputSize, inputSize)
   self.weightB = torch.Tensor(outputSize, inputSize)
   self.weightOrg = torch.Tensor(outputSize, inputSize)
   self.groups = groups or 1
   self.bias = torch.Tensor(outputSize*self.groups)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize*self.groups)
   self:reset()
   -- should nil for serialization, the reset will still work
   self.reset = nil
end

function SplitBinaryLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2)/self.groups)
   end
   --print(stdv)
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-1, 1)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-1, 1)
      self.bias:uniform(-stdv, stdv)
   end

   return self
end

function SplitBinaryLinear:binarized()
  self.weightOrg:copy(self.weight)
  self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)
  self.weightB:round():mul(2):add(-1)
  return  self.weightB
end

function SplitBinaryLinear:updateOutput(input)
  self.weightB = self:binarized()
  self.weight:copy(self.weightB)
  parent.updateOutput(self,input)
  self.weight:copy(self.weightOrg)
  return self.output
end

function SplitBinaryLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.weight:copy(self.weightB)
      parent.updateGradInput(self,input, gradOutput)
      self.weight:copy(self.weightOrg)
      return self.gradInput
   end

end

function SplitBinaryLinear:accGradParameters(input, gradOutput, scale)
  parent.accGradParameters(self,input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
SplitBinaryLinear.sharedAccUpdateGradParameters = SplitBinaryLinear.accUpdateGradParameters

