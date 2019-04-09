-- This is Fully-Connected layer with binary weights
local BinaryLinear, parent = torch.class('BinaryLinear', 'nn.Linear')

function BinaryLinear:__init(inputSize, outputSize)
   local delayedReset = self.reset
   self.reset = function() end
   parent.__init(self, inputSize, outputSize)
   self.reset = delayedReset

   self.weight = torch.Tensor(outputSize, inputSize)
   self.weightB = torch.Tensor(outputSize, inputSize)
   self.weightOrg = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self:reset()
   -- should nil for serialization, the reset will still work
   self.reset = nil
end

function BinaryLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
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

function BinaryLinear:binarized()
  self.weightOrg:copy(self.weight)
  self.weightB:copy(self.weight):add(1):div(2):clamp(0,1)
  self.weightB:round():mul(2):add(-1)
  return  self.weightB
end

function BinaryLinear:updateOutput(input)
  self.weightB = self:binarized()
  self.weight:copy(self.weightB)
  parent.updateOutput(self,input)
  self.weight:copy(self.weightOrg)
  return self.output
end

function BinaryLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      self.weight:copy(self.weightB)
      parent.updateGradInput(self,input, gradOutput)
      self.weight:copy(self.weightOrg)
      return self.gradInput
   end
end

function BinaryLinear:accGradParameters(input, gradOutput, scale)
  parent.accGradParameters(self,input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
BinaryLinear.sharedAccUpdateGradParameters = BinaryLinear.accUpdateGradParameters


function BinaryLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
