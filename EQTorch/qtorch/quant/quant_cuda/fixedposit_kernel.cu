#include "quant_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define FP16_LIMB_SIZE 16
#define FP32_LIMB_SIZE 32
#define FP16_TYPE uint16_t
#define FP32_TYPE uint32_t //**edited

// defining int32_constants as an array containing 16 elements of type uint32_t
__constant__ uint32_t int32_constants[16];
// same for this case
__constant__ uint64_t int64_constants[2];

// table lookup data, normally we use 5 bit (32 values). But this array size needs to be hardcoded before compilation.
// Use 128 to support up to 7 bits usigned table lookup.(the lookup assume 1 bit automaticcally used for sign => max 8 bits supported by this table)
__constant__ float table_lookup[128];
__constant__ float rounding_hint[128];

#define SIGN_MASK 0x80000000                     // 8000 => 1000 0000 0000 0000(16 bits)
#define FLOAT_SIGN_MASK 0x80000000           // => 1000 0000 0000 0000 0000 0000 0000 0000 (32 bits)
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF     // => 1 0 bit 31 1 bits
#define SECOND_BIT_MASK 0x40000000               // 4000 => 0100 0000 0000 0000
#define FIXED_POSIT_INF 0x00000000               // defining the inf for posit
#define FIXED_POSIT_LIMB_ALL_BITS_SET 0xffffffff // all bits set for a posit
#define SINGLE_PRECISION_BIAS 127            // bias for the single precision number
#define FLOAT_SIZE 32                        // single precision
#define FLOAT_EXPONENT_MASK 0x7f800000       //=> 0111 1111 1000 0000 0000 0000 0000 0000
#define FLOAT_FRACTION_MASK 0x007fffff       //=> 0000 0000 0111 1111 1111 1111 1111 1111
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000   //=> 0000 0000 1000 0000 0000 0000 0000 0000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8 // name defined
#define TEMP_TYPE uint64_t                     // defining the temporary type to be unint64_t
#define UNSIGNED_LONG_LONG_SIZE 64             // name defined
#define EDP_ACC_SIZE 63                        // setting the accumulator size
#define FIXED_POSIT_EXP_SHIFT 41               // 64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000         // 0111 1111 1000 0000 0000 0000 0000 0000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9 // name defined
#define FIXED_POSIT_LENGTH_PLUS_ONE 33 /

#define GET_MAX(a, b)           \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
    })

#define _G_INFP 2147483648

union Bits
{
    float f;
    int32_t si;
    uint32_t ui;
};
typedef FP16_TYPE fp16;
typedef FP32_TYPE fp32;
#define _G_FPOSIT_SHIFT_AMOUNT int32_constants[0]
#define _G_MAXREALFP int32_constants[1]
#define _G_MINREALFP int32_constants[2]
#define FPOSIT_EXTRA_BITS_SHIFT int32_constants[3]
#define _G_USEED int32_constants[4]
#define _G_USEED_ZEROS int32_constants[5]
#define FPOSIT_EXPONENT_MASK int32_constants[6]

#define _G_MAXREAL_INT int32_constants[7]
#define _G_MINREAL_INT int32_constants[8]
#define _G_NBITS int32_constants[9]
#define _G_ESIZE int32_constants[10]

#define _G_RSIZE int32_constants[11]
#define FPOSIT_REGIME_MASK int32_constants[12]
#define FPOSIT_FRACTION_MASK int32_constants[13]
#define _G_FSIZE int32_constants[14]
#define _FP_REGIME_BIAS int32_constants[15]
// define regime size

#define FPOSIT_EXTRA_BITS_MASK int64_constants[0]
#define FPOSIT_HALFWAY_BIT_MASK int64_constants[1]

void generate_fixedposit_constants(int nsize, int es, int rf, uint32_t *int32_constants, uint64_t *int64_constants)
{
    // local vars have the same name as global constant vars, confusing but less likely error can happen here.
    // ugly but it's the straightforward conversion from the original #define macroes;
    // todo: make this one less messy

    _G_NBITS = nsize;
    _G_ESIZE = es;
    _G_RSIZE = rf;
    _G_FSIZE = nsize - es - rf - 1;

    if (nsize <= 32 && rf <= 10)
    {
        _G_FPOSIT_SHIFT_AMOUNT = FP32_LIMB_SIZE - nsize;
        // _G_MAXREALFP = ((1 << (nsize - 1)) - 1) << _G_FPOSIT_SHIFT_AMOUNT;
        _G_MAXREALFP = ((((1<<_G_RSIZE)-2)<<_G_ESIZE)|((1<<_G_ESIZE)-1))<<(_G_FSIZE+_G_FPOSIT_SHIFT_AMOUNT);
        // _G_MINREALFP = 1<<_G_FPOSIT_SHIFT_AMOUNT;
        _G_MINREALFP = 1<<(_G_ESIZE+_G_FSIZE+_G_FPOSIT_SHIFT_AMOUNT);
        FPOSIT_EXTRA_BITS_SHIFT = UNSIGNED_LONG_LONG_SIZE - nsize + 1;
        _G_USEED = 1 << (1 << es);
        _G_USEED_ZEROS = (1 << es);
        FPOSIT_EXPONENT_MASK = _G_USEED_ZEROS - 1;
        _FP_REGIME_BIAS = 0;

        _G_MAXREAL_INT = ((((rf-2)<<es)+((1<<es)-1))+SINGLE_PRECISION_BIAS)<<FLOAT_EXPONENT_SHIFT;
        // _G_MAXREAL_INT = ((_G_USEED_ZEROS * (nsize - 2)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // this is the maximum float integer that can be represented by the fixed posit.
        _G_MINREAL_INT = ((-1* (_G_RSIZE-1) * (_G_USEED_ZEROS))+SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // _G_MINREAL_INT = ((_G_USEED_ZEROS * (2 - nsize)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // this is the minimum float integer that can be represented by the fixed posit.
        FPOSIT_REGIME_MASK = ((1 << rf) - 1) << (_G_FSIZE + _G_ESIZE);
        FPOSIT_FRACTION_MASK = ((1 << _G_FSIZE) - 1);

        FPOSIT_EXTRA_BITS_MASK = (1UL << (UNSIGNED_LONG_LONG_SIZE - nsize)) - 1;
        FPOSIT_HALFWAY_BIT_MASK = 1UL << (UNSIGNED_LONG_LONG_SIZE - nsize);
    }
    else
    {
        printf("unexpected fposit config\n");
        exit(1);
    }
};

__device__ __inline__ float fixedp16tofp32_gpu(fp32 Fp)
{
  union Bits v;
  v.si = 0;
	// get sign
	bool sign = Fp & SIGN_MASK;//
	Fp = (Fp ^ -sign) + sign;//
  // printf("the current float formed is: %d \n", v.si);
	// get the regime sign
	bool regime_sign = Fp & SECOND_BIT_MASK;//

	// get regime
	v.ui = Fp << 1;
	//int regime_length = (__builtin_clz(v.ui) & -!regime_sign) + (__builtin_clz(~v.ui) & -regime_sign);
  // printf("the current float formed is: %d \n", v.si);
	int m;
	  if(regime_sign)
	    m = (__builtin_clz(~v.ui));
	  else
	    m = (__builtin_clz(v.ui));
    // int regime_length = _G_RSIZE;
	int k2es = (m - regime_sign) << _G_ESIZE;
  // printf("the m is: %d \n", m);
	k2es = (k2es ^ -regime_sign) + regime_sign;
  // printf("the current float formed is: %d \n", v.si);
	// assemble
	v.ui <<= (_G_RSIZE);
	v.ui >>= (FLOAT_SIGN_PLUS_EXP_LENGTH - _G_ESIZE);
	// printf("the k2es is: %d \n", k2es);
  v.ui += ((SINGLE_PRECISION_BIAS - k2es) << FLOAT_EXPONENT_SHIFT);
  // printf("the current float formed is: %d \n", v.si);
	v.si ^= (FLOAT_INF ^ v.si) & -(Fp == _G_INFP);
	v.si ^= (0 ^ v.si) & -(Fp == 0);

	v.ui |= (sign << FLOAT_SIGN_SHIFT);
	return v.f;
}

__device__ __inline__ fp32 fp32tofixedp16_gpu(float f)
{
  fp32 Fp = 0;
  union Bits v;
  v.f = f;
  bool sign = v.ui & FLOAT_SIGN_MASK;
  v.ui &= 0x7FFFFFFF;
  // printf("the current input value is: %d \n", v.ui);
#ifdef FLOAT_ROUNDING
	uint16_t roundSign = sign << 15;
	if(v.ui > _G_MAXREAL_INT)
		return _G_INFP | roundSign;
	if(v.ui < _G_MINREAL_INT)
		return 0;
#endif
  // printf("the current output value is: %d \n", Fp);
  Fp ^= (Fp ^_G_MAXREALFP) & -(v.si >= _G_MAXREAL_INT);
  // printf("the current output value is: %d \n", Fp);
  Fp ^= (Fp ^ _G_INFP) & -(v.si >= FLOAT_INF);
  // printf("the current output value is: %d \n", Fp);
  Fp ^= (Fp ^ _G_MINREALFP) & -(v.si != 0 && v.si <= _G_MINREAL_INT);
  // printf("the current output value is: %d \n", Fp);
  // min posit exponent in 16, 3 is 112
  // therefore all the float subnormals will be handled
  // in the previous if statement

  // get exponent sign
  bool exp_sign = !(v.ui >> FLOAT_EXP_SIGN_SHIFT);

  //get regime and exponent
  uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS);
  // printf("the current exponent value is: %d \n", exp);
  TEMP_TYPE regime_and_exp = (((1 << ((exp >> _G_ESIZE) + 1)) - 1) << (_G_RSIZE-((exp >> _G_ESIZE) + 1)+_G_ESIZE)) | (exp & FPOSIT_EXPONENT_MASK);
  //if exponent is negative
  // printf("the current regime and exp value is: %d \n", regime_and_exp);
  regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign);// >> ((exp_sign & !((exp & FPOSIT_EXPONENT_MASK))) & (bool) exp);
  // printf("the current regime and exp value is: %d \n", regime_and_exp);
  // int regime_and_exp_length = (exp >> _G_ESIZE) + 2 + _G_ESIZE - ((exp_sign & !((exp & FPOSIT_EXPONENT_MASK))) & (bool) exp);
  int regime_and_exp_length = _G_ESIZE+_G_RSIZE;

  //assemble
  regime_and_exp <<= (UNSIGNED_LONG_LONG_SIZE - regime_and_exp_length);
  regime_and_exp |= ((TEMP_TYPE) (v.ui & FLOAT_FRACTION_MASK) << (FIXED_POSIT_EXP_SHIFT - regime_and_exp_length));
  fp32 temp_p = (regime_and_exp >> FPOSIT_EXTRA_BITS_SHIFT);

  //round
  temp_p += (bool) (regime_and_exp & FPOSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & FPOSIT_EXTRA_BITS_MASK));
  // printf("the current output value is: %d \n", Fp);
 if (_G_NBITS != 32)
    // {printf("the current output value is: %d \n", Fp);
    temp_p <<= _G_FPOSIT_SHIFT_AMOUNT;
    // }
    
  Fp ^= (temp_p ^ Fp) & -((v.si < _G_MAXREAL_INT) & (v.si > _G_MINREAL_INT));

  Fp = (Fp ^ -sign) + sign;
  // printf("the final output value is: %d \n", Fp);
  return Fp;
}


__global__ void fixed_posit_kernel_nearest(float *input, float *output, float scale, size_t input_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < input_size)
    {
        float temp_input = input[index] * scale;

        fp32 temp = fp32tofixedp16_gpu(temp_input);
        temp_input = fixedp16tofp32_gpu(temp);

        output[index] = temp_input / scale;
    }
}

// __device__ float new_format_quantize_nearest(float input)
// {
//     float constants[32] = {1.0 / 65536, 1.0 / 32768, 1.0 / 16384, 1.0 / 8192, 1.0 / 4096, 1.0 / 2048, 1.0 / 1024, 1.0 / 512, 1.0 / 256, 1.0 / 128,
//                            3.0 / 256, 1.0 / 64, 5.0 / 256, 3.0 / 128, 7.0 / 256, 1.0 / 32, 9.0 / 256, 5.0 / 128, 3.0 / 64, 7.0 / 128,
//                            1.0 / 16, 9.0 / 128, 5.0 / 64, 3.0 / 32, 7.0 / 64, 1.0 / 8, 9.0 / 64, 3.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 2, 1.0};
//     float result = 0.0;
//     if (input != 0.0)
//     {

//         float min_abs_err = 1e5;
//         float min_constant = 0.0;
//         for (int i = 0; i < 32; i++)
//         {
//             float abs_err = fabsf(constants[i] - fabsf(input));
//             if (abs_err < min_abs_err)
//             {
//                 min_abs_err = abs_err;
//                 min_constant = constants[i];
//             }
//         }

//         if (input < 0)
//             result = -min_constant;
//         else
//             result = min_constant;
//     }

//     return result;
// }

// __device__ float act_format_quantize_nearest(float input)
// {

//     float constants[32] = {1.0 / 4096, 1.0 / 2048, 1.0 / 1024, 1.0 / 512, 1.0 / 256, 1.0 / 128, 1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 3.0 / 16,
//                            1.0 / 4, 5.0 / 16, 3.0 / 8, 7.0 / 16, 1.0 / 2, 9.0 / 16, 5.0 / 8, 3.0 / 4, 7.0 / 8, 1.0, 9.0 / 8, 5.0 / 4, 3.0 / 2,
//                            7.0 / 4, 2.0, 9.0 / 4, 3.0, 4.0, 6.0, 8.0, 16.0};
//     float result = 0.0;
//     if (input != 0.0)
//     {

//         float min_abs_err = 1e5;
//         float min_constant = 0.0;
//         for (int i = 0; i < 32; i++)
//         {
//             float abs_err = fabsf(constants[i] - fabsf(input));
//             if (abs_err < min_abs_err)
//             {
//                 min_abs_err = abs_err;
//                 min_constant = constants[i];
//             }
//         }

//         if (input < 0)
//             result = -min_constant;
//         else
//             result = min_constant;
//     }

//     return result;
// }

// __device__ float configurable_table_quantize_nearest(float input, int table_size)
// {

//     float result = 0.0;
//     if (input != 0.0)
//     {

//         float min_abs_err = 1e5;
//         float min_constant = 0.0;
//         for (int i = 0; i < table_size; i++)
//         {
//             float abs_err = fabsf(table_lookup[i] - fabsf(input));
//             if (abs_err < min_abs_err)
//             {
//                 min_abs_err = abs_err;
//                 min_constant = table_lookup[i];
//             }
//         }

//         if (input < 0)
//             result = -min_constant;
//         else
//             result = min_constant;
//     }

//     return result;
// }

// __device__ float configurable_table_quantize_rounding_hint(float input, int table_size)
// {

//     float result = 0.0;
//     if (input != 0.0)
//     {
//         float min_constant = 0.0;
//         for (int i = 0; i < table_size; i++)
//         {
//             // float abs_err = fabs(constants[i] - fabs(input));
//             if (fabsf(input) > rounding_hint[i])
//                 min_constant = table_lookup[i];
//             // printf( " %f %f %f \n", input , rounding_hint[i] , min_constant);
//         }

//         if (input < 0)
//             result = -min_constant;
//         else
//             result = min_constant;
//     }

//     return result;
// }

// // template <typename scalar_t>
// __global__ void newformat_kernel_nearest(float *input, float *output, float scale, size_t input_size)
// {
//     const int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < input_size)
//     {
//         float temp_input = input[index] * scale;

//         temp_input = new_format_quantize_nearest(temp_input);

//         output[index] = temp_input / scale;
//     }
// }

// __global__ void actformat_kernel_nearest(float *input, float *output, float scale, size_t input_size)
// {
//     const int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < input_size)
//     {
//         float temp_input = input[index] * scale;

//         temp_input = act_format_quantize_nearest(temp_input);

//         output[index] = temp_input / scale;
//     }
// }

// __global__ void configurable_table_kernel_nearest(float *input, float *output, float scale, size_t input_size, size_t table_size)
// {
//     const int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < input_size)
//     {
//         float temp_input = input[index] * scale;

//         temp_input = configurable_table_quantize_nearest(temp_input, table_size);

//         output[index] = temp_input / scale;
//     }
// }

// __global__ void configurable_quantize_kernel_rounding_hint(float *input, float *output, float scale, size_t input_size, size_t table_size)
// {
//     const int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < input_size)
//     {
//         float temp_input = input[index] * scale;

//         temp_input = configurable_table_quantize_rounding_hint(temp_input, table_size);

//         output[index] = temp_input / scale;
//     }
// }


void fixed_posit_kernel_nearest_wrapper(float *__restrict__ a,
                                  float *o, int size, int nsize, int es, int rf, float scale, int blockNums, int blockSize)
{

    uint32_t int32_constants_host[16];
    uint64_t int64_constants_host[2];
    generate_fixedposit_constants(nsize, es, rf, int32_constants_host, int64_constants_host);

    cudaMemcpyToSymbol(int32_constants, &int32_constants_host[0], 16 * sizeof(uint32_t), 0);
    cudaMemcpyToSymbol(int64_constants, &int64_constants_host[0], 2 * sizeof(uint64_t), 0);

    fixed_posit_kernel_nearest<<<blockNums, blockSize>>>(a,
                                                   o,
                                                   scale,
                                                   size);
}

// void newformat_kernel_nearest_wrapper(float *__restrict__ a,
//                                       float *o, int size, float scale, int blockNums, int blockSize)
// {

//     newformat_kernel_nearest<<<blockNums, blockSize>>>(a,
//                                                        o,
//                                                        scale,
//                                                        size);
// }

// void actformat_kernel_nearest_wrapper(float *__restrict__ a,
//                                       float *o, int size, float scale, int blockNums, int blockSize)
// {

//     actformat_kernel_nearest<<<blockNums, blockSize>>>(a,
//                                                        o,
//                                                        scale,
//                                                        size);
// }

// void configurable_quantize_kernel_nearest_wrapper(float *__restrict__ a,
//                                                   float *o,
//                                                   /*table lookup data*/
//                                                   float *constants,
//                                                   int table_size, int size, float scale, int blockNums, int blockSize)
// {

//     cudaMemcpyToSymbol(table_lookup, &constants[0], table_size * sizeof(float), 0);
//     configurable_table_kernel_nearest<<<blockNums, blockSize>>>(a,
//                                                                 o,
//                                                                 scale,
//                                                                 size,
//                                                                 table_size);
// }

// void configurable_quantize_kernel_rounding_hint_wrapper(float *__restrict__ a,
//                                                         float *o,
//                                                         /*table lookup data*/
//                                                         float *constants,
//                                                         float *round_hints,
//                                                         int table_size, int size, float scale, int blockNums, int blockSize)
// {

//     cudaMemcpyToSymbol(table_lookup, &constants[0], table_size * sizeof(float), 0);
//     cudaMemcpyToSymbol(rounding_hint, &round_hints[0], table_size * sizeof(float), 0);

//     configurable_quantize_kernel_rounding_hint<<<blockNums, blockSize>>>(a,
//                                                                          o,
//                                                                          scale,
//                                                                          size,
//                                                                          table_size);
// }