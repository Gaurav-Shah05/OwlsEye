# OwlsEye
 Owlseye is a custom and real-time implementation of Video Instance Segmentation designed to accomodate low-light conditions. It's optimized for edge applications through OpenVino 8-bit Quantization-Aware Training, Async Pipelining, Brightness Verfication, etc. to give a performance upto 30 FPS at 720p resolution. Tested on Intel Nezha board with LogiTech C270 Webcam.

## NOTE
This Repo is incomplete, but it contains the [jupyter notebook](yolov8-instance-segmentation.ipynb) used for downloading, quantizing, and verifying the accuracy for the yolov8-nano instance segmentation model that it used in the real-time code for live video instance segmentation.

- To understand how the quantized model is being used in the final runfile, You can refer to [live_optimized.py](live_optimized.py) file.

I will update the repository with the complete code soon.

## Results

The videos showing the owlseye performance are in the `.\Results` folder.
1. [Original Captured Video](Results\lowlight_noenhance.avi)
2. [Direct Segmentation using YoloV8 on the Captured Video](Results\lowlight_original.avi)
3. [OwlsEye](Results\lowlight_zerodce_filter.avi)