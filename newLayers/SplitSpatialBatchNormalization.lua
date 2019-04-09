--This is batchnormalization layer with new parameter 'groups' for debugging 
-- It targets 4-dim output of spatial convolution
local BN, parent = torch.class('SplitSpatialBatchNormalization', 'SplitBatchNormalization')

BN.__version = 2

-- expected dimension of input
BN.nDim = 4