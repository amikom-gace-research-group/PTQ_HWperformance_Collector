import os
import re
import time
import psutil
import subprocess
from timeit import default_timer as timer
import numpy as np
from PIL import Image
import warnings
import threading
import multiprocessing
from platform import uname
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
                    with threading.Lock():
                        self._list.append(cpu_)
            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except Exception as e:
            print(f"Error in CPU measurement : {e}")
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
                    with threading.Lock():
                        self._list.append(power_)
            self.event.clear()
            res = sum(self._list) / len(self._list)
            self.result = res, self._list
        except:
            self.result = 0, self._list

    def stop(self):
        self.event.set()

class GPUMem(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []

    def run(self):
        try:
            while not self.event.is_set():
                pattern = re.compile(rf"{os.getpid()}\s+(\d+)K")
                with open("/sys/kernel/debug/nvmap/iovmm/maps", "r") as fp:
                    content = fp.read()
                    match = pattern.search(content)
                    if match:
                        with threading.Lock():
                            self._list.append(float(match.group(1))/1024)
                    
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

    def _clear_cache(self, passwd):
        subprocess.run(["sudo", "-S", "sync"], input=passwd, universal_newlines=True)
        subprocess.run(["sudo", "-S", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"], input=passwd, universal_newlines=True)

    def _process_memory(self):
        process = psutil.Process()
        mem_info = process.memory_full_info()
        return mem_info 

    def _jstat_start(self, passwd):
        subprocess.run(["sudo", "-S", "tegrastats", "--interval", "10", "--start", "--logfile", f"tegrastats_{os.getpid()}.txt"], input=passwd, universal_newlines=True)

    def _jstat_stop(self, type, passwd):
        subprocess.run(["sudo", "-S", "tegrastats", "--stop"], input=passwd, universal_newlines=True)
        out = open(f"tegrastats_{os.getpid()}.txt", 'r')
        lines = out.read().split('\n')
        entire_gpu = []
        entire_power = []
        entire_power_gpu = []
        entire_power_cpu = []
        freq = 0
        try:
            for line in lines:
                if '4.9.337-tegra' == uname().release:
                    pattern_pow = r"POM_5V_IN (\d+)/(\d+)"
                    match_pow = re.search(pattern_pow, line)
                    if match_pow:
                        power_ = match_pow.group(2)
                        entire_power.append(float(power_))
                    pattern_pow_cpu = r"POM_5V_CPU (\d+)/(\d+)"
                    match_pow_cpu = re.search(pattern_pow_cpu, line)
                    if match_pow_cpu:
                        power_cpu_ = match_pow_cpu.group(2)
                        entire_power_cpu.append(float(power_cpu_))
                elif '5.10.104-tegra' == uname().release:
                    pattern_pow = r"VDD_IN (\d+)mW/(\d+)mW"
                    match_pow = re.search(pattern_pow, line)
                    if match_pow:
                        power_ = match_pow.group(2)
                        entire_power.append(float(power_))
                    pattern_pow_cpu = r"VDD_SOC (\d+)mW/(\d+)mW"
                    match_pow_cpu = re.search(pattern_pow_cpu, line)
                    if match_pow_cpu:
                        power_cpu_ = match_pow_cpu.group(2)
                        entire_power_cpu.append(float(power_cpu_))
                if type == 'gpu':
                    pattern_gpu = r"GR3D_FREQ (\d+)%@(\d+)"
                    match_gpu = re.search(pattern_gpu, line)
                    if match_gpu:
                        gpu_ = match_gpu.group(1)
                        freq = match_gpu.group(2)
                        entire_gpu.append(float(gpu_))
                    if '4.9.337-tegra' == uname().release:
                        pattern_pow_gpu = r"POM_5V_GPU (\d+)/(\d+)"
                        match_pow_gpu = re.search(pattern_pow_gpu, line)
                        if match_pow_gpu:
                            power_gpu_ = match_pow_gpu.group(2)
                            entire_power_gpu.append(float(power_gpu_))
                    elif '5.10.104-tegra' == uname().release:
                        pattern_pow_gpu = r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW"
                        match_pow_gpu = re.search(pattern_pow_gpu, line)
                        if match_pow_gpu:
                            power_gpu_ = match_pow_gpu.group(2)
                            entire_power_gpu.append(float(power_gpu_))
            entire_gpu_ = [num for num in entire_gpu if num > 0.0]
            entire_power_ = [num for num in entire_power if num > 0.0]
            entire_power_gpu_ = [num for num in entire_power_gpu if num > 0.0]
            entire_power_cpu_ = [num for num in entire_power_cpu if num > 0.0]
            result_gpu = sum(entire_gpu_) / len(entire_gpu_) if entire_gpu_ else 0
            result_power = sum(entire_power_) / len(entire_power_) if entire_power_ else 0
            result_power_gpu = sum(entire_power_gpu_) / len(entire_power_gpu_) if entire_power_gpu_ else 0
            result_power_cpu = sum(entire_power_cpu_) / len(entire_power_cpu_) if entire_power_cpu_ else 0
        except:
            result_gpu = 0
            result_power = 0
            result_power_cpu = 0
            result_power_gpu = 0
            entire_gpu_ = entire_gpu
            entire_power_ = entire_power
            entire_power_cpu_ = entire_power_cpu
            entire_power_gpu_ = entire_power_gpu

        return result_gpu, result_power, freq, result_power_cpu, result_power_gpu, entire_gpu_,  entire_power_, entire_power_cpu_, entire_power_gpu_
    
    # @profile
    def tflite_benchmark(self, iterations, dev_type, threads, passwd):
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=self._graph_path, num_threads=threads)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]

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
        
        def inference_worker(input_queue, output_queue):
            hwperfs = {'Task Time':[], 'The Num. of Task':[], 'Output':[]}
            con_start = timer()
            while True:
                input_data = input_queue.get()
                if input_data is None:
                    break

                interpreter.set_tensor(input_details['index'], input_data)

                hwperf = []

                # Run inference.
                cpu = CPU()
                cpu.start()
                if 'tegra' in uname().release:
                    self._jstat_start(passwd)
                elif check_ina219():
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
                cpu_freq = psutil.cpu_freq().current
                cpu.stop()
                cpu.join()
                if 'tegra' in uname().release:
                    power, _, power_cpu = self._jstat_stop(dev_type, passwd)[1:4]
                elif check_ina219():
                    ina.stop()
                    ina.join()
                    power = float(ina.result[0])
                else:
                    power = 0
                    power_cpu = 0
                cpu_percent = float(cpu.result[0])
                hwperf.append([round(elapsed, 2), round(cpu_percent, 2), [round(mem_res.rss/1024**2, 2), round(mem_res.swap/1024**2, 2)], round(power, 2), round(power_cpu, 2), float(cpu_freq)])
                if 'tegra' in uname().release:
                    subprocess.check_output(f'rm tegrastats_{os.getpid()}.txt', shell=True)
                # clear cache
                self._clear_cache(passwd)
            
            con_time = ((timer()- con_start) * 1000)
            hwperfs["Task Time"].append(con_time)
            hwperfs["Output"].append(hwperf)
            hwperfs["The Num. of Task"].append(len(hwperf))
            output_queue.put(hwperfs)

        # Initialize queues for communication between main process and worker processes
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        # Create and start worker processes
        num_processes = psutil.cpu_count()
        processes = []

        for _ in range(num_processes):
            p = multiprocessing.Process(target=inference_worker, args=(input_queue, output_queue))
            p.start()
            processes.append(p)
        
        total_hwperf = []

        # Push input data to the input queue
        for _ in range(iterations + 1):
            input_queue.put(input_data)

        for _ in range(num_processes):
            input_queue.put(None)  # Signal workers to exit

        # Retrieve results from the output queue
        for _ in range(iterations + 1):
            res = output_queue.get()
            total_hwperf.append(res)

        # Wait for all worker processes to finish
        for p in processes:
            p.join()

        return total_hwperf
    
    def tensorrt_benchmark(self, iterations, dev_type, passwd):
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner
        from polygraphy.logger import G_LOGGER
        G_LOGGER.module_severity = 50
        load_engine = EngineFromBytes(BytesFromPath(self._graph_path))
        with TrtRunner(load_engine) as runner:
            input_metadata = runner.get_input_metadata()
            img = Image.open(self._img).resize((224, 224))
            frame = np.array(img, dtype=input_metadata["input_1"].dtype) / 255.0
            input_data = np.expand_dims(frame, axis=0).astype(input_metadata["input_1"].dtype)

            def inference_worker(input_queue, output_queue):
                hwperfs = {'Task Time':[], 'Num. of Tasks':[], 'Output':[]}
                con_start = timer()
                while True:
                    input_data = input_queue.get()
                    if input_data is None:
                        break
                    hwperf = []
                    runner.activate()
                    self._jstat_start(passwd)
                    cpu = CPU()
                    gmem = GPUMem()
                    cpu.start()
                    gmem.start()
                    time.sleep(2)
                    runner.infer(feed_dict={'input_1': input_data})

                    # retrieve the results
                    mem_res = self._process_memory()
                    gpu, power, gpu_freq, power_cpu, power_gpu = self._jstat_stop(dev_type, passwd)[0:5]
                    elapsed = runner.inference_time * 1000
                    if elapsed < 1000:
                        time.sleep((2000-elapsed)/1000)
                    cpu_freq = psutil.cpu_freq().current
                    cpu.stop()
                    gmem.stop()
                    cpu.join()
                    gmem.join()
                    cpu_percent = float(cpu.result[0])
                    gpu_mem = float(gmem.result[0])

                    hwperf.append([round(elapsed, 2), round(cpu_percent, 2), [round(mem_res.rss/1024**2, 2), round(mem_res.swap/1024**2, 2)], round(gpu, 2), round(power, 2), round(power_cpu, 2), round(power_gpu, 2), round(gpu_mem, 2), round(float(cpu_freq), 2), round(float(gpu_freq), 2)])
                    runner.deactivate()
                    subprocess.check_output(f'rm tegrastats_{os.getpid()}.txt', shell=True)
                    # clear cache
                    self._clear_cache(passwd)

                con_time = ((timer()- con_start) * 1000)
                hwperfs["Task Time"].append(con_time)
                hwperfs["Output"].append(hwperf)
                hwperfs["Num. of Tasks"].append(len(hwperf))
                output_queue.put(hwperfs)
            
            # Initialize queues for communication between main process and worker processes
            input_queue = multiprocessing.Queue()
            output_queue = multiprocessing.Queue()

            # Create and start worker processes
            num_processes = psutil.cpu_count()
            processes = []

            for _ in range(num_processes):
                p = multiprocessing.Process(target=inference_worker, args=(input_queue, output_queue))
                p.start()
                processes.append(p)
            
            total_hwperf = []

            # Push input data to the input queue
            for _ in range(iterations + 1):
                input_queue.put(input_data)

            for _ in range(num_processes):
                input_queue.put(None)  # Signal workers to exit

            # Retrieve results from the output queue
            for _ in range(iterations + 1):
                res = output_queue.get()
                total_hwperf.append(res)

            # Wait for all worker processes to finish
            for p in processes:
                p.join()

            return total_hwperf

def check_ina219():
    try:
        import ina219
        ina = ina219.INA219(0.1)
        voltage = ina.voltage()
        # If the script reaches here without errors, the INA219 is connected
        return True
    except:
        return False

### RUN CODE FUNC ###
        
def main_tflite(model : str, iterations : int, dev_type : str, threads, passwd : str):
    setup = GetLatency(graph_path=model, img='dlperf_meter/assets/flower.jpg')
    hwperf = setup.tflite_benchmark(iterations, dev_type, threads, passwd)

    return hwperf

def main_tensorrt(model : str, iterations : int, dev_type : str, passwd : str):
    setup = GetLatency(graph_path=model, img='dlperf_meter/assets/flower.jpg')
    hwperf = setup.tensorrt_benchmark(iterations, dev_type, passwd)
    
    return hwperf

if __name__ == '__main__':
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model', required=True)
    parser.add_argument('--type', help='device types', required=True)
    parser.add_argument('--threads', help='num_threads (just for tflite)', default=None)
    parser.add_argument('--iterations', help='how many model runs (auto add warmup once)', default=1)
    config = configparser.ConfigParser()
    config.read("._config.ini")
    _passwd = config.get("Credentials", "password", raw=True)
    args = parser.parse_args()
    
    if 'cpu' in args.type:
        data = main_tflite(args.model, int(args.iterations), args.type, (int(args.threads) if isinstance(args.threads, int) else None), _passwd)
    elif 'gpu' in args.type:
        data = main_tensorrt(args.model, int(args.iterations), args.type, _passwd)
    
    print(data)
