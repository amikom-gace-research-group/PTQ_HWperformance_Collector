import os
import re
import gc
import time
import psutil
import subprocess
from timeit import default_timer as timer
import numpy as np
from PIL import Image
import warnings
import threading
warnings.filterwarnings('ignore')

### FUNCTION ###
class CPU(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                output = subprocess.check_output(['pidstat', '-p', str(os.getpid()), '1', '1'])
                cpu_ = float(output.splitlines()[-2].split()[-3])
                if cpu_ > 0.0:
                    self._list.append(cpu_)
            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list

    def stop(self):
        self.event.set()

class INAEXT(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                from ina219 import INA219
                ina = INA219(0.1)
                ina.configure()
                power_ = ina.power()
                if power_ > 0.0:
                    self._list.append(power_)
            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list

    def stop(self):
        self.event.set()

class GetLatency:
    def __init__(self, graph_path='', img=''):
        """
        @params:
        graph_path: graph file (.tflite)
        num_threads: The number of threads to use for running TFLite interpreter. default is 4
        gpu: bool True to enable GPU mode benchmarks (tflite_benchmark function only)
        img: image for testing by interpreter_benchmark function (not return accuracy, just to fill input tensor).
        """
        self._graph_path = graph_path
        self._img = img
        self.gpu = True

    def _process_memory(self):
        process = psutil.Process()
        mem_info = process.memory_full_info()
        return mem_info 

    def _jstat_start(self, passwd):
        subprocess.check_output(f'echo {passwd} | sudo -S tegrastats --interval 10 --start --logfile test.txt', shell=True)

    def _jstat_stop(self, passwd):
        subprocess.check_output(f'echo {passwd} | sudo -S tegrastats --stop', shell=True)
        out = open("test.txt", 'r')
        lines = out.read().split('\n')
        entire_gpu = []
        entire_power = []
        try:
            for line in lines:
                pattern = r"GR3D_FREQ (\d+)%"
                match = re.search(pattern, line)
                if match:
                    gpu_ = match.group(1)
                    entire_gpu.append(float(gpu_))
                pattern = r"POM_5V_IN (\d+)/(\d+)"
                match = re.search(pattern, line)
                if match:
                    power_ = match.group(2)
                    entire_power.append(float(power_))
            entire_gpu_ = [num for num in entire_gpu if num > 0.0]
            entire_power_ = [num for num in entire_power if num > 0.0]
            result_gpu = sum(entire_gpu_) / len(entire_gpu_)
            result_power = sum(entire_power_) / len(entire_power_)
        except:
            result_gpu = 0
            result_power = 0
            entire_gpu_ = entire_gpu
            entire_power_ = entire_power

        return result_gpu, result_power, entire_gpu_,  entire_power_
    
    # @profile
    def tflite_benchmark(self, iterations, type, threads, passwd):
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=self._graph_path, num_threads=int(threads))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_index = interpreter.get_output_details()[0]["index"]

        height = input_details['shape'][1]
        width = input_details['shape'][2]

        img = Image.open(self._img).resize((width, height))

        # Convert image to NumPy array
        img_array = np.array(img, dtype=input_details["dtype"]) / 255.0

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            img_array = img_array / input_scale + input_zero_point

        # add N dim
        input_data = np.expand_dims(img, axis=0).astype(input_details["dtype"])

        interpreter.set_tensor(input_details['index'], input_data)

        hwperf = []

        for i in np.arange(iterations+1):
            # Run inference.
            thread = CPU()
            thread.start()
            if 'jnano' in type:
                self._jstat_start(passwd)
            elif 'rasp' in type:
                ina = INAEXT()
                ina.start()
            time.sleep(2)
            start = timer()
            interpreter.invoke()
            mem_res = self._process_memory()
            end = timer()
            elapsed = ((end - start) * 1000)
            if elapsed < 1000:
                time.sleep((2000-elapsed)/1000)
            thread.stop()
            thread.join()
            if 'jnano' in type:
                power = float(self._jstat_stop(passwd)[1])
            elif 'rasp' in type:
                ina.stop()
                ina.join()
                power = float(ina.result[0])
            cpu_percent = float(thread.result[0])
            hwperf.append([round(elapsed, 2), round(cpu_percent, 2), [round(mem_res.rss/1024**2, 2), round(mem_res.pss/1024**2, 2), round(mem_res.uss/1024**2, 2)], round(power, 2)])

        # clear session / past cache
        tf.keras.backend.clear_session()
        gc.collect()

        return hwperf
    
    def tensorrt_benchmark(self, iterations, passwd):
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        from polygraphy.logger import G_LOGGER
        G_LOGGER.module_severity = 50
        load_engine = EngineFromBytes(BytesFromPath(self._graph_path))
        with TrtRunner(load_engine) as runner:
            input_metadata = runner.get_input_metadata()
            img = Image.open(self._img).resize((224, 224))
            frame = np.array(img, dtype=input_metadata['dtype']) / 255.0
            input_data = np.expand_dims(frame, axis=0).astype(input_metadata['dtype'])

            hwperf = []

            for i in np.arange(iterations+1):
                runner.activate()
                # create the threads
                self._jstat_start(passwd)
                thread = CPU()
                thread.start()
                time.sleep(2)
                runner.infer(feed_dict={'input_1': input_data})

                # retrieve the results
                mem_res = self._process_memory()
                gpu, power = self._jstat_stop(passwd)[0:2]
                elapsed = runner.inference_time / 1000
                if elapsed < 1000:
                    time.sleep((2000-elapsed)/1000)
                thread.stop()
                thread.join()
                cpu_percent = float(thread.result[0])

                hwperf.append([round(elapsed, 2), round(cpu_percent, 2), [round(mem_res.rss/1024**2, 2), round(mem_res.pss/1024**2, 2), round(mem_res.uss/1024**2, 2)], round(gpu, 2), round(power, 2)])

            # deactivate and clear cache
            runner.deactivate()
            gc.collect()

        return hwperf

### RUN CODE FUNC ###
        
def main_tflite(model, iterations, type, threads, passwd):
    setup = GetLatency(graph_path=model, img='dlperf_meter/assets/flower.jpg')
    hwperf = setup.tflite_benchmark(iterations, type, threads, passwd)

    return hwperf

def main_tensorrt(model, iterations, passwd):
    setup = GetLatency(graph_path=model, img='dlperf_meter/assets/flower.jpg')
    hwperf = setup.tensorrt_benchmark(iterations, passwd)
    
    return hwperf

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model', required=True)
    parser.add_argument('--type', help='device types', required=True)
    parser.add_argument('--threads', help='num_threads (just for tflite)', default=1)
    parser.add_argument('--iterations', help='how many model runs (auto add warmup once)', default=1)
    parser.add_argument('--passwd', help='user password', required=True)
    args = parser.parse_args()
    
    if 'tflite' in args.type:
        data = main_tflite(args.model, int(args.iterations), args.type, args.threads, args.passwd)
    elif 'trt' in args.type:
        data = main_tensorrt(args.model, int(args.iterations), args.passwd)
        subprocess.check_output('rm test.txt', shell=True)
    
    print(data)