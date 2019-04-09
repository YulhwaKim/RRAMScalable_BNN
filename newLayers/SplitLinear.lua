local SplitLinear, parent = torch.class('nn.SplitLinear', 'nn.Module')

function SplitLinear:__init(inputSize, outputSize, groups, bias)
   parent.__init(self)
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize)
   self.groups = groups or 1
   self.c = math.sqrt(2.0/self.weight:size(2)) / math.sqrt(self.groups)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   if bias then
      self.bias = torch.Tensor(outputSize*self.groups)
      self.gradBias = torch.Tensor(outputSize*self.groups)
   end
   self:reset()
end

function SplitLinear:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function SplitLinear:reset(stdv)
   self.weight:copy(torch.randn(self.weight:size())*self.c)
   self.bias:fill(0)
   return self
end

function SplitLinear:updateAddBuffer(input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function SplitLinear:updateOutput(input)
   local inChannel = self.weight:size(2) / self.groups
   local outChannel = self.weight:size(1)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1)*self.groups)
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      for i = 0, self.groups-1 do
         local temp_output = self.output[{{1+outChannel*i, outChannel*(i+1)}}]
         local temp_weight = self.weight[{{},{1+inChannel*i, inChannel*(i+1)}}]
         local temp_input = input[{{1+inChannel*i, inChannel*(i+1)}}]
         temp_output:addmv(1, temp_weight, temp_input)
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1)*self.groups)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self:updateAddBuffer(input)
      for i = 0, self.groups-1 do
         local temp_output = self.output[{{},{1+outChannel*i, outChannel*(i+1)}}]
         local temp_input = input[{{},{1+inChannel*i, inChannel*(i+1)}}]
         local temp_weight = self.weight[{{},{1+inChannel*i, inChannel*(i+1)}}]
         temp_output:addmm(0, temp_output, 1, temp_input, temp_weight:t())
         if self.bias then temp_output:addr(1, self.addBuffer, self.bias[{{1+outChannel*i, outChannel*(i+1)}}]) end
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function SplitLinear:updateGradInput(input, gradOutput)
   if self.gradInput then
      local inChannel = self.weight:size(2) / self.groups
      local outChannel = self.weight:size(1)

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         for i = 0, self.groups-1 do
            local temp_gradOutput = gradOutput[{{1+outChannel*i, outChannel*(i+1)}}] 
            local temp_weight = self.weight[{{},{1+inChannel*i, inChannel*(i+1)}}]
            local temp_gradInput = self.gradInput[{{1+inChannel*i, inChannel*(i+1)}}]
            temp_gradInput:addmv(0, 1, temp_weight:t(), temp_gradOutput)
         end
      elseif input:dim() == 2 then
         for i = 0, self.groups-1 do
            local temp_gradOutput = gradOutput[{{},{1+outChannel*i, outChannel*(i+1)}}] 
            local temp_weight = self.weight[{{},{1+inChannel*i, inChannel*(i+1)}}]
            local temp_gradInput = self.gradInput[{{},{1+inChannel*i, inChannel*(i+1)}}]
            temp_gradInput:addmm(0, 1, temp_gradOutput, temp_weight)
         end
      end
      return self.gradInput
   end
end

function SplitLinear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local inChannel = self.weight:size(2) / self.groups
   local outChannel = self.weight:size(1)
   if input:dim() == 1 then
      for i = 0, self.groups-1 do
         local temp_gradOutput = gradOutput[{{1+outChannel*i, outChannel*(i+1)}}] 
         local temp_gradWeight = self.gradWeight[{{},{1+inChannel*i, inChannel*(i+1)}}]
         local temp_input = input[{{1+inChannel*i, inChannel*(i+1)}}]
         
         temp_gradWeight:addr(scale, temp_gradOutput, temp_input)
         if self.bias then self.gradBias:add(scale, temp_gradOutput) end
      end
   elseif input:dim() == 2 then
      for i = 0, self.groups-1 do
         local temp_gradOutput = gradOutput[{{},{1+outChannel*i, outChannel*(i+1)}}] 
         local temp_gradWeight = self.gradWeight[{{},{1+inChannel*i, inChannel*(i+1)}}]
         local temp_input = input[{{},{1+inChannel*i, inChannel*(i+1)}}]
         
         temp_gradWeight:addmm(scale, temp_gradOutput:t(), temp_input)
      end
      if self.bias then
         self:updateAddBuffer(input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

function SplitLinear:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end

function SplitLinear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function SplitLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d), groups: %d', self.weight:size(2), self.weight:size(1), self.groups) ..
      (self.bias == nil and ' without bias' or '')
end