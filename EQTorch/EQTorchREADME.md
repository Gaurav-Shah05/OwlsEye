# EQTorch

The framework fully supports training neural networks with fixed posit quantization. This feature allows researchers and developers to leverage the parallisation and highly optimised multiplication of floating points in CUDA to simulate fixed posit arithmetic in neural network computations.

Install or use the framework by running the following command:

```bash
pip install -e ./
```
Once installed, you can import the tools from `qtorch` as follows:

```python
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from qtorch import Posit, FixedPosit
```
## Changes made

### 5.1 Addition of Test Components

We added a new `FixedPosit` folder within the `test` directory. This folder contains:

1. **`test.cpp` file**: This file includes several essential functions:
   - Generate fixed posit constants function  
   - Fixedp16tofp32 conversion function  
   - Fp32tofixedp16 conversion function  
   - Additional supporting components  

### 5.2 Core Functionality Implementation

A significant addition was made to the `quant_cuda` folder:

1. **`fixedposit_kernel.cu` file**: This file is central to our GPU based fixed posit implementation and includes:
   - Generate fixed posit constants function  
   - CUDA-based Fixedp16tofp32 conversion function  
   - CUDA-based Fp32tofixedp16 conversion function  
   - Fixed posit kernel nearest function and its wrapper  

In addition to these major additions, various general edits throughout the framework were made to ensure the integration and operation of the new Fixed Posit functionality.  These changes collectively enable the framework to support Fixed Posit quantization, providing a robust foundation for neural network training with this number system.

## Citation

This work builds upon and extends the concepts introduced in QPyTorch+. 

```bibtex
@inproceedings{ho2022qtorch+,
  title={Qtorch+: Next Generation Arithmetic for Pytorch Machine Learning},
  author={Ho, Nhut-Minh and De Silva, Himeshi and Gustafson, John L and Wong, Weng-Fai},
  booktitle={Conference on Next Generation Arithmetic},
  pages={31--49},
  year={2022},
  organization={Springer}
}
