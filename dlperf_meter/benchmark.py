import os
import re
import json
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
            pass

    def stop(self):
        self.event.set()

class INAEXT(threading.Thread):
    from ina219 import INA219
    from ina219 import DeviceRangeError
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self.SHUNT_OHMS = 0.1
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                ina = INA219(self.SHUNT_OHMS)
                ina.configure()
                power_ = ina.power()
                if power_ > 0.0:
                    self._list.append(power_)
            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list
            pass

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

    def _gpu_start(self):
        subprocess.check_output('tegrastats --interval 10 --start --logfile test.txt', shell=True)

    def _gpu_stop(self):
        subprocess.check_output('tegrastats --stop', shell=True)
        out = open("test.txt", 'r')
        lines = out.read().split('\n')
        entire = []
        try:
            for line in lines:
                pattern = r"GR3D_FREQ (\d+)%"
                match = re.search(pattern, line)
                if match:
                    gpu_ = match.group(1)
                    entire.append(float(gpu_))
            entire_ = [num for num in entire if num > 0.0]
            result = sum(entire_) / len(entire_)
        except Exception as e:
            result = 0
            entire_ = entire
            pass

        return result, entire_
    
    # @profile
    def tflite_benchmark(self, type):
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=self._graph_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32
        uint8_model = input_details[0]['dtype'] == np.uint8

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        img = Image.open(self._img).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference.
        thread = CPU()
        thread.start()
        if 'jnano' in type:
            jetson = jtop(interval=0.01)
            jetson.start()
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
            power = jetson.power['avg']
            jetson.stop()
        elif 'rasp' in type:
            ina.stop()
            ina.join()
            power = float(ina.result[0])
        cpu_percent = float(thread.result[0])
        return elapsed, [round(mem_res.rss/1024**2, 2), round(mem_res.pss/1024**2, 2), round(mem_res.uss/1024**2, 2)], cpu_percent, power
    
    def tensorrt_benchmark(self):
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        from jtop import jtop
        from polygraphy.logger import G_LOGGER
        G_LOGGER.module_severity = 50
        load_engine = EngineFromBytes(BytesFromPath(self._graph_path))
        with TrtRunner(load_engine) as runner:
            img = Image.open(self._img).resize((224, 224))
            frame = np.array(img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(frame, axis=0).astype(np.float32)
            self._gpu_start()

            # create the threads
            thread = CPU()
            thread.start()
            jetson = jtop(interval=0.01)
            jetson.start()
            time.sleep(2)
            start_time = timer()
            outputs = runner.infer(feed_dict={'input_1': input_data})
            mem_res = self._process_memory()
            end_time = timer()

        # retrieve the results
        gpu = float(self._gpu_stop()[0])
        elapsed = ((end_time - start_time) * 1000)
        if elapsed < 1000:
            time.sleep((2000-elapsed)/1000)
        thread.stop()
        thread.join()
        power = jetson.power['avg']
        jetson.stop()
        cpu_percent = float(thread.result[0])
        return [round(mem_res.rss/1024**2, 2), round(mem_res.pss/1024**2, 2), round(mem_res.uss/1024**2, 2)], gpu, elapsed, cpu_percent, power

### RUN CODE FUNC ###
        
def main_tflite(model, type):
    setup = GetLatency(graph_path=model, img='latency_meter/assets/flower.jpg')
    lat, mem, cpu, power = setup.tflite_benchmark(type)

    res = [round(lat, 2), round(float(cpu), 2), mem, power]

    return res

def main_tensorrt(model):
    setup = GetLatency(graph_path=model, img='latency_meter/assets/flower.jpg')
    mem, gpu, lat, cpu, power = setup.tensorrt_benchmark()
    
    res = [round(lat, 2), round(gpu, 2), round(cpu, 2), mem, power]
    return res

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model', required=True)
    parser.add_argument('--type', help='device types', required=True)
    parser.add_argument('--iteration', help='how many model runs (auto add warmup once)', default=1)
    parser.add_argument('--passwd', help='user password', required=True)
    args = parser.parse_args()
    data = []
    for i in range(int(args.iteration)+1):
        os.system(f"echo {args.passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")
        if 'tflite' in args.type:
            d = main_tflite(args.model, args.type)
            data.append(d)
        elif 'trt' in args.type:
            d = main_tensorrt(args.model)
            data.append(d)
            subprocess.check_output('rm test.txt', shell=True)
    print(data)
    os.system(f"echo {args.passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")

