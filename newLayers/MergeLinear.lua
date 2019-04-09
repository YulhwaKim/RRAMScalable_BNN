local MergeLinear,parent = torch.class('MergeLinear', 'nn.Module')


function MergeLinear:__init(groups)
   parent.__init(self)
   self.groups = groups or 1
 end


function MergeLinear:updateOutput(input)
    local nChannel = input:size(2)/self.groups
    local firstGroup = input:narrow(2,1,nChannel)
    self.output:resizeAs(firstGroup)
    self.output:copy(firstGroup)
    for i = 1, self.groups -1 do
      self.output:add(input:narrow(2, 1+nChannel*i, nChannel))
    end
   return self.output
end

function MergeLinear:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:repeatTensor(1,self.groups)
  return self.gradInput
end
