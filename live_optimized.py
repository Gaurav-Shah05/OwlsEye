import cv2
import numpy as np
import time
import openvino as ov
from pathlib import Path
from ultralytics import YOLO
import asyncio
import queue
import threading
import logging

date_time = time.strftime("%d-%H%M")
Path(f'live_run/{date_time}').mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=f'live_run/{date_time}/log.txt', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',)

class BrightnessChecker:
    def __init__(self, threshold=50):
        self.threshold = threshold

    def is_low_light(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < self.threshold

class VideoEnhancer:
    def __init__(self, model_path):
        core = ov.Core()
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_shape = self.model.inputs[0].shape
        self.target_size = (self.input_shape[3], self.input_shape[2])

    def enhance_frame(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        self.infer_request.infer(inputs={self.model.inputs[0].get_any_name(): preprocessed_frame})
        enhanced_frame = self.infer_request.get_output_tensor(0).data
        return self.postprocess_frame(enhanced_frame, frame.shape[:2])

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.target_size)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return np.expand_dims(frame, axis=0)

    def postprocess_frame(self, enhanced_frame, original_shape):
        enhanced_frame = np.transpose(enhanced_frame[0], (1, 2, 0))
        enhanced_frame = (enhanced_frame * 255.0).astype(np.uint8)
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        if enhanced_frame.shape[:2] != original_shape:
            enhanced_frame = cv2.resize(enhanced_frame, (original_shape[1], original_shape[0]))
        return enhanced_frame

class VideoProcessor:
    def __init__(self, enhancer_model_path, seg_model_path, buffer_size=5):
        self.enhancer = VideoEnhancer(enhancer_model_path)
        self.seg_model = YOLO(seg_model_path)
        self.brightness_checker = BrightnessChecker()
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.processed_frames = queue.Queue(maxsize=buffer_size)
        self.skip_frames = 10
        self.frame_count = 0
        self.last_segmented_frame = None
        self.enhancement_on = False

    def denoise_frame(self, frame):
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def process_frame(self, frame):
        self.frame_count += 1
        do_segment = self.frame_count % self.skip_frames == 0

        if self.brightness_checker.is_low_light(frame):
            frame = self.enhancer.enhance_frame(frame)
            self.enhancement_on = True
        else:
            self.enhancement_on = False

        frame = self.denoise_frame(frame)

        if do_segment:
            seg_result = self.seg_model(frame)
            self.last_segmented_frame = seg_result[0].plot()
        
        return self.last_segmented_frame if self.last_segmented_frame is not None else frame

    def start_processing(self):
        while True:
            frame = self.buffer.get()
            if frame is None:
                break
            processed_frame = self.process_frame(frame)
            self.processed_frames.put(processed_frame)
            self.buffer.task_done()

def capture_frames(cap, processor):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not processor.buffer.full():
            processor.buffer.put(frame)
        else:
            _ = processor.buffer.get()  # Remove oldest frame if buffer is full
            processor.buffer.put(frame)

async def main():
    processor = VideoProcessor(
        enhancer_model_path="optimized_models/zerodce/optmized_model/quantized_model.xml",
        seg_model_path="./models/yolov8n-seg.pt"
    )
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)

    # Start the processing thread
    processing_thread = threading.Thread(target=processor.start_processing)
    processing_thread.start()

    # Start the capture thread
    capture_thread = threading.Thread(target=capture_frames, args=(cap, processor))
    capture_thread.start()

    fps_list = []
    last_time = time.time()
    frame_count = 0

    while True:
        if not processor.processed_frames.empty():
            processed_frame = processor.processed_frames.get()
            
            current_time = time.time()
            frame_count += 1
            
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                fps_list.append(fps)
                if len(fps_list) > 30:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                
                # cv2.putText(processed_frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Enhancement: {'ON' if processor.enhancement_on else 'OFF'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                
                frame_count = 0
                last_time = current_time

            cv2.imshow("Processed Video", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            await asyncio.sleep(0.001)  # Small delay to prevent busy waiting

    # Cleanup
    processor.buffer.put(None)  # Signal the processing thread to stop
    processing_thread.join()
    capture_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())