RRAMScalable_BNN
=================================================
This is the complete training example of [Neural Network-Hardware Co-design for Scalable RRAM-based BNN Accelerators](https://arxiv.org/abs/1811.02187)
on following datasets: Cifar10, MNIST

Although in-memory analog computation is promising for MV multiplication and NN acceleration, the overhead of analog-digital interfaces limit the efficiency of analog computation.

[Neural Network-Hardware Co-design for Scalable RRAM-based BNN Accelerators](https://arxiv.org/abs/1811.02187) presented a methodology to split inputs of a Binary Neural Network (BNN) to fit each sub-network on a RRAM array. This way, we can accelerate BNN with typical memory configuration with 1-bit word line driver and 1-bit sense-amp. 

## Requirements 
This code is implemented on top of [BinaryNet](https://github.com/itayhubara/BinaryNet) and has all the same requirements.

## Training
Start training using:
```lua
th Main_Cifar10.lua -network spCifar10_Model -arraySize 256 -nGPU 2
th Main_MNIST.lua -network spMNIST_Model -arraySize 256
```

## Trained models
The inference accuracy on the Cifar10, MNIST are as follows:

| Network (arraySize) | MNIST | CIFAR-10 |
| 		:---:  		  | :---: |  :---:   |
| Baseline (-)		  | 98.6 | 88.6 |
| Split  (512)        | 98.7 | 88.3 |
| Split  (256)        | 98.8 | 88.3 |
| Split  (128)        | 98.6 | 87.5 |

Note that the trained model used default options except 'network', 'arraySize', and 'nGPU'.
