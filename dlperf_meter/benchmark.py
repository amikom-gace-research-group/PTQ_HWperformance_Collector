import os
import re
import sys
import time
import psutil
import subprocess
from timeit import default_timer as timer
from jtop import jtop
import numpy as np
from PIL import Image
import warnings
import threading
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
                        self._list.append([cpu_, psutil.cpu_freq().current])
            self.event.clear()
            list_res = []
            list_freq = []
            for r in self._list:
                list_res.append(r[0])
                list_freq.append(r[1])
            res = sum(list_res) / len(list_res)
            freq = sum(list_freq) / len(list_freq)
            self.result = res, freq
        except Exception as e:
            #print(f"Error in CPU measurement : {e}")
            self.result = 0, 0

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
            self.result = res
        except:
            self.result = 0

    def stop(self):
        self.event.set()

class GPUMem(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.result = None
        self.event = threading.Event()
        self._list = []
        self._freq = []

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

# class JSTAT(threading.Thread):
#     def __init__(self, dev_type):
#         threading.Thread.__init__(self)
#         self.result = None
#         self.event = threading.Event()
#         self._dev_type = dev_type
#         self._l = []

#     def run(self):
#         list_tot_pow_5v_in = []
#         list_avg_pow_5v_cpu = []
#         list_avg_pow_5v_gpu = []
#         list_gpu_load = []
#         list_gpu_mem = []
#         list_freq_cur = []
#         try:
#             while not self.event.is_set():
#                 with jtop(interval=0.01) as jetson:
#                     if jetson.ok():
#                         l1 = jetson.power
#                         l2 = jetson.gpu
#                         l3 = jetson.processes
#                         self._l.append([l1, l2, l3])
#             self.event.clear()
#             for ltw in self._l:
#                 tot_pow_5v_in = ltw[0]['tot']['avg']
#                 list_tot_pow_5v_in.append(tot_pow_5v_in)
#                 if '4.9.337-tegra' == uname().release:
#                     avg_pow_5v_cpu = ltw[0]['rail']['POM_5V_CPU']['avg']
#                     list_avg_pow_5v_cpu.append(avg_pow_5v_cpu)
#                     if self._dev_type == "gpu":
#                         avg_pow_5v_gpu = ltw[0]['rail']['POM_5V_GPU']['avg']
#                         gpu_load = ltw[1]['gpu']['status']['load']
#                         freq_cur = ltw[1]['gpu']['freq']['cur']
#                         for mgpu in ltw[2]:
#                             if mgpu[0] == os.getpid():
#                                 list_gpu_mem.append(mgpu[8])
#                         list_avg_pow_5v_gpu.append(avg_pow_5v_gpu)
#                         list_gpu_load.append(gpu_load)
#                         list_freq_cur.append(freq_cur)
#             entire_gpu_ = [num for num in list_gpu_load if num > 0.0]
#             entire_power_ = [num for num in list_tot_pow_5v_in if num > 0.0]
#             entire_power_gpu_ = [num for num in list_avg_pow_5v_gpu if num > 0.0]
#             entire_power_cpu_ = [num for num in list_avg_pow_5v_cpu if num > 0.0]
#             entire_gpu_mem_ = [num for num in list_gpu_mem if num > 0.0]
#             avg_freq = sum(list_freq_cur) / len(list_freq_cur) if list_freq_cur else 0
#             result_gpu = sum(entire_gpu_) / len(entire_gpu_) if entire_gpu_ else 0
#             result_power = sum(entire_power_) / len(entire_power_) if entire_power_ else 0
#             result_power_gpu = sum(entire_power_gpu_) / len(entire_power_gpu_) if entire_power_gpu_ else 0
#             result_power_cpu = sum(entire_power_cpu_) / len(entire_power_cpu_) if entire_power_cpu_ else 0
#             result_gpu_mem = sum(entire_gpu_mem_) / len(entire_gpu_mem_) if entire_gpu_mem_ else 0
#             self.result = result_gpu, result_power, avg_freq, result_power_cpu, result_power_gpu, result_gpu_mem
#         except Exception as e:
#             print(e)
#             result_gpu = 0
#             result_power = 0
#             result_power_cpu = 0
#             result_power_gpu = 0
#             avg_freq = 0
#             result_gpu_mem = 0
#             self.result = result_gpu, result_power, avg_freq, result_power_cpu, result_power_gpu, result_gpu_mem
    
#     def stop(self):
#         self.event.set()

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

    def _process_memory(self):
        process = psutil.Process()
        mem_info = process.memory_full_info()
        return mem_info 

    def _jstat_start(self, passwd=""):
        subprocess.run(["sudo", "-S", "tegrastats", "--interval", "10", "--start", "--logfile", f"tegrastats_{os.getpid()}.txt"], input=passwd, universal_newlines=True)

    def _jstat_stop(self, dev_type="", passwd=""):
        subprocess.run(["sudo", "-S", "tegrastats", "--stop"], input=passwd, universal_newlines=True)
        try:
            file_path = f"tegrastats_{os.getpid()}.txt"
            if os.path.exists(file_path):
                out = open(file_path, 'r')
            else:
                raise FileNotFoundError
            lines = out.read().split('\n')
            
            entire_gpu = []
            entire_power = []
            entire_power_gpu = []
            entire_power_cpu = []
            entire_freq_gpu = []
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
                if dev_type == 'gpu':
                    pattern_gpu = r"GR3D_FREQ (\d+)%@(\d+)"
                    match_gpu = re.search(pattern_gpu, line)
                    if match_gpu:
                        gpu_ = match_gpu.group(1)
                        freq = match_gpu.group(2)
                        entire_freq_gpu.append(float(freq))
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
            entire_gpu_ = [num for num in entire_gpu if num > 20.0]
            entire_power_ = [num for num in entire_power if num > 0.0]
            entire_power_gpu_ = [num for num in entire_power_gpu if num > 0.0]
            entire_power_cpu_ = [num for num in entire_power_cpu if num > 0.0]
            result_freq = sum(entire_freq_gpu) / len(entire_freq_gpu) if entire_freq_gpu else 0
            result_gpu = sum(entire_gpu_) / len(entire_gpu_) if entire_gpu_ else 0
            result_power = sum(entire_power_) / len(entire_power_) if entire_power_ else 0
            result_power_gpu = sum(entire_power_gpu_) / len(entire_power_gpu_) if entire_power_gpu_ else 0
            result_power_cpu = sum(entire_power_cpu_) / len(entire_power_cpu_) if entire_power_cpu_ else 0
            
            return result_gpu, result_power, result_freq, result_power_cpu, result_power_gpu
        except Exception as e:
            print("tegrastats error :", e)
            result_gpu = 0
            result_power = 0
            result_power_cpu = 0
            result_power_gpu = 0
            result_freq = 0

            return result_gpu, result_power, result_freq, result_power_cpu, result_power_gpu
    
    # @profile
    def tflite_benchmark(self, iterations, threads, passwd):
        try:
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

            interpreter.set_tensor(input_details['index'], input_data)

            hwperf = []

            for i in np.arange(iterations+1):
                # Run inference.
                start = timer()
                interpreter.invoke()
                end = timer()
                elapsed = ((end - start) * 1000)

                hwperf.append([round(elapsed, 2)])
                # clear cache
                sync_command = ["sudo", "-S", "sync"]
                subprocess.run(sync_command, input=passwd, universal_newlines=True)
                drop_caches_command = ["sudo", "-S", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
                subprocess.run(drop_caches_command, input=passwd, universal_newlines=True)
                
            return hwperf
        except Exception as e:
            #print(f"Failure bencmarking, error {e}")
            pass
    
    def tensorrt_benchmark(self, iterations, passwd):
        try:
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

                hwperf = []

                for i in np.arange(iterations+1):
                    runner.infer(feed_dict={'input_1': input_data})

                    # retrieve the results
                    elapsed = runner.inference_time * 1000

                    hwperf.append([round(elapsed, 2)])
                    
                    # clear cache
                    sync_command = ["sudo", "-S", "sync"]
                    subprocess.run(sync_command, input=passwd, universal_newlines=True)
                    drop_caches_command = ["sudo", "-S", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
                    subprocess.run(drop_caches_command, input=passwd, universal_newlines=True)
                
            return hwperf
        except Exception as e:
            #print(f"Failure bencmarking, error {e}")
            pass

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
        
def main(model : str, iterations : int, dev_type : str, threads, passwd : str):#, process_id, result_list):
    hwperf = None
    setup = GetLatency(graph_path=model, img='dlperf_meter/assets/flower.jpg')
    if dev_type == "cpu":
        hwperf = setup.tflite_benchmark(iterations, threads, passwd)
    elif dev_type == "gpu":
         hwperf = setup.tensorrt_benchmark(iterations, passwd)

    #result_list[process_id] = hwperf
    return hwperf

def run_multiprocessing(num_threads, model, iterations, dev_type, threads, passwd):
    import concurrent.futures

    mes = GetLatency()
    mes._jstat_start(passwd=passwd)
    cpu = CPU()
    cpu.start()
    gmem = GPUMem()
    gmem.start()
    time.sleep(2)
    start = timer()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(main, model, iterations, dev_type, threads, passwd) for _ in range(num_threads)]
        try:
            deadline = concurrent.futures.wait(futures, timeout=1860)
            results = [completed_future.result() for completed_future in deadline.done]
        except concurrent.futures.TimeoutError:
            print(f"Timeout, {model} task concurrency level {num_threads} is too long (+30 minutes)")
            sys.exit(0)
    elapsed = (timer() - start) * 1000
    mem_res = mes._process_memory().rss/1024**2
    gpu, power, gpu_freq, power_cpu, power_gpu = mes._jstat_stop(passwd=passwd, dev_type=dev_type)
    if elapsed < 1000:
        time.sleep((2000-elapsed)/1000)
    cpu.stop()
    cpu.join()
    gmem.stop()
    gmem.join()
    cpu_result = cpu.result
    cpu_percent = cpu_result[0]
    cpu_freq = cpu_result[1]
    gmem_result = gmem.result[0]
    subprocess.run(['sudo', '-S', 'rm', f'tegrastats_{os.getpid()}.txt'], input=passwd, universal_newlines=True)

    return [round(elapsed, 2), round(power, 2), round(cpu_freq, 2), round(cpu_percent, 2), round(power_cpu, 2), round(gpu_freq, 2), round(gpu, 2), round(power_gpu, 2), round(gmem_result, 2), round(mem_res, 2), results]

if __name__ == '__main__':
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the detection model', required=True)
    parser.add_argument('--type', help='device types', required=True)
    parser.add_argument('--threads', help='num_threads (just for tflite)', default=None)
    parser.add_argument('--iterations', help='how many model runs (auto add warmup once)', default=1)
    parser.add_argument('--concurrent', help='how many concurrent runs', default=1, type=int)
    config = configparser.ConfigParser()
    config.read("._config.ini")
    _passwd = config.get("Credentials", "password", raw=True)
    args = parser.parse_args()
    
    data = run_multiprocessing(args.concurrent, args.model, int(args.iterations), args.type, (int(args.threads) if isinstance(args.threads, int) else None), _passwd)
    #data = main(args.model, int(args.iterations), args.type, (int(args.threads) if isinstance(args.threads, int) else None), _passwd)
    new_data = []
    new_data.append(data[:-1])
    for datum in data[-1]:
        if datum != None:
            new_data.extend(datum)
    print(new_data)
