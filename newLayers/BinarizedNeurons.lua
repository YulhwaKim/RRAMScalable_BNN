-- This layer binarize input to [-1, 1]
local BinarizedNeurons,parent = torch.class('BinarizedNeurons', 'nn.Module')

function BinarizedNeurons:__init()
   parent.__init(self)
end

function BinarizedNeurons:updateOutput(input)
    self.output = input:ge(0):typeAs(input):mul(2):add(-1)
   return self.output
end

function BinarizedNeurons:updateGradInput(input, gradOutput)
		-- If you want to use different memory space for the gradInput and gradOutput, 
		-- then turn on this
        -- self.gradInput:resizeAs(gradOutput)
        -- self.gradInput:copy(gradOutput)

        -- This option uses the same memory space for gradInput and gradOutput
        self.gradInput = gradOutput
   return self.gradInput
end
