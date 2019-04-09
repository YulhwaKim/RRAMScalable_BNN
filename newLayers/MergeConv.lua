local MergeConv,parent = torch.class('MergeConv', 'nn.Module')


function MergeConv:__init(groups)
   parent.__init(self)
   self.groups = groups
 end


function MergeConv:updateOutput(input)
    local nChannel = input:size(2)/self.groups
    local firstGroup = input:narrow(2,1,nChannel)
    self.output:resizeAs(firstGroup)
    self.output:copy(firstGroup)
    for i = 1, self.groups -1 do
      self.output:add(input:narrow(2, 1+nChannel*i, nChannel))
    end
   return self.output
end

function MergeConv:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput:repeatTensor(1,self.groups,1,1)
  return self.gradInput
end
