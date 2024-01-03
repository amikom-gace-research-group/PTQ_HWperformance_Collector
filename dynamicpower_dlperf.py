import os
import ast
import pandas as pd
import numpy as np
import time
import psutil
import logging
import subprocess
from platform import uname
from dlperf_meter.benchmark import check_ina219

def run(passwd : str, model_path : str, dev_type : str, threads, iterations : int):
    model_name = os.path.basename(model_path)
    print("Model : ", model_name)
    if 'gpu' in dev_type:
        template = {'Model':[model_name], 'Model Size (MB)':[get_size(model_path, 'mb')], 'J_Clock':[jetson_stat()[0]], 'J_NVP':[jetson_stat()[1]], 'CPU Cores':[psutil.cpu_count()], 'Warmup-CPU Freq (MHz)':[], 'Warmup-GPU Freq (MHz)':[], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (MB)':[], 'Warmup-Mem Swap Usage (MB)':[], 'Warmup-Mem GPU Usage (MB)':[], 'Warmup-Power (mW)':[], 'Warmup-Power CPU (mW)':[], 'Warmup-Power GPU (mW)':[], 'Warmup-GPU Usage (%)':[]}
    else:
        template = {'Model':[model_name], 'Model Size (MB)':[get_size(model_path, 'mb')], 'Num Threads':[threads], 'CPU Cores':[psutil.cpu_count()], 'Warmup-CPU Freq (MHz)':[], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (MB)':[], 'Warmup-Mem Swap Usage (MB)':[]}
    print('iterations :', iterations)
    print('Jetson Mode :', jetson_stat())
    benchmark_command = [
    "sudo", "python3", "dlperf_meter/benchmark.py",
    "--model", model_path,
    "--type", dev_type,
    "--iterations", str(iterations)
    ]
    # Include threads in the command if it's not None
    if threads is not None:
        benchmark_command.extend(["--threads", threads])
    cmd = subprocess.run(benchmark_command, input=passwd, stdout=subprocess.PIPE, universal_newlines=True).stdout
    data = ast.literal_eval(cmd)
    for idx, j in enumerate(data):
        if 'cpu' in dev_type:
            if idx == 0:
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-CPU Usage (%)'].append(float(j[1]))
                template['Warmup-Mem RSS Usage (MB)'].append(float(j[2][0]))
                template['Warmup-Mem Swap Usage (MB)'].append(float(j[2][1]))
                template['Warmup-CPU Freq (MHz)'].append(float(j[5]))
                if 'tegra' in uname().release or check_ina219():
                    if 'Warmup-Power (mW)' not in template:
                        template['Warmup-Power (mW)'] = []
                    if 'tegra' in uname().release:
                        if 'J_Clock' not in template:
                            template['J_Clock'] = []
                        if 'J_NVP' not in template:
                            template['J_NVP'] = []
                        if 'Warmup-Power CPU (mW)' not in template:
                            template['Warmup-Power CPU (mW)'] = []
                        template['J_Clock'].append(jetson_stat()[0])
                        template['J_NVP'].append(jetson_stat()[1])
                        template['Warmup-Power CPU (mW)'].append(float(j[4]))
                    template['Warmup-Power (mW)'].append(float(j[3]))
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (iter-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (iter-{idx}) (MB)'] = []
                    template[f'Memory Swap Usage (iter-{idx}) (MB)'] = []
                    template[f'CPU Freq (iter-{idx}) (MHz)'] = []
                    if 'tegra' in uname().release or check_ina219():
                        if f'Power (iter-{idx}) (mW)' not in template:
                            template[f'Power (iter-{idx}) (mW)'] = []
                        if 'tegra' in uname().release:
                            if f'Power CPU (iter-{idx}) (mW)' not in template:
                                template[f'Power CPU (iter-{idx}) (mW)'] = []
                            template[f'Power CPU (iter-{idx}) (mW)'].append(float(j[4]))
                        template[f'Power (iter-{idx}) (mW)'].append(float(j[3]))
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'CPU Usage (iter-{idx}) (%)'].append(float(j[1]))
                template[f'Memory RSS Usage (iter-{idx}) (MB)'].append(float(j[2][0]))
                template[f'Memory Swap Usage (iter-{idx}) (MB)'].append(float(j[2][1]))
                template[f'CPU Freq (iter-{idx}) (MHz)'].append(float(j[5]))
        elif 'gpu' in dev_type:
            if idx == 0:
                template['Warmup-GPU Freq (MHz)'].append(j[9])
                template['Warmup-CPU Freq (MHz)'].append(j[8])
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-GPU Usage (%)'].append(float(j[3]))
                template['Warmup-CPU Usage (%)'].append(float(j[1]))
                template['Warmup-Mem RSS Usage (MB)'].append(float(j[2][0]))
                template['Warmup-Mem Swap Usage (MB)'].append(float(j[2][1]))
                template['Warmup-Mem GPU Usage (MB)'].append(float(j[7]))
                template['Warmup-Power (mW)'].append(float(j[4]))
                template['Warmup-Power CPU (mW)'].append(float(j[5]))
                template['Warmup-Power GPU (mW)'].append(float(j[6]))
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'CPU Freq (iter-{idx}) (MHz)'] = []
                    template[f'GPU Freq (iter-{idx}) (MHz)'] = []
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (iter-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (iter-{idx}) (MB)'] = []
                    template[f'Memory Swap Usage (iter-{idx}) (MB)'] = []
                    template[f'Memory GPU Usage (iter-{idx}) (MB)'] = []
                    template[f'GPU Usage (iter-{idx}) (%)'] = []
                    template[f'Power (iter-{idx}) (mW)'] = []
                    template[f'Power CPU (iter-{idx}) (mW)'] = []
                    template[f'Power GPU (iter-{idx}) (mW)'] = []
                template[f'CPU Freq (iter-{idx}) (MHz)'].append(float(j[8]))
                template[f'GPU Freq (iter-{idx}) (MHz)'].append(float(j[9]))
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'GPU Usage (iter-{idx}) (%)'].append(float(j[3]))
                template[f'CPU Usage (iter-{idx}) (%)'].append(float(j[1]))
                template[f'Memory RSS Usage (iter-{idx}) (MB)'].append(float(j[2][0]))
                template[f'Memory Swap Usage (iter-{idx}) (MB)'].append(float(j[2][1]))
                template[f'Memory GPU Usage (iter-{idx}) (MB)'].append(float(j[7]))
                template[f'Power (iter-{idx}) (mW)'].append(float(j[4]))
                template[f'Power CPU (iter-{idx}) (mW)'].append(float(j[5]))
                template[f'Power GPU (iter-{idx}) (mW)'].append(float(j[6]))

    df = pd.DataFrame(template)
    output_path=f'dynamicpower_dlperf_{dev_type}_{uname().release}.csv'
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False)
    else: df.to_csv(output_path)
    print('='*25)

def main(passwd : str, model_path : str, dev_type: str, threads, iterations : int):
    if '5.10.104-tegra' == uname().release:
        for id in np.arange(0, 9):
            for clk in [True, False]:
                j_mode(id, clk)
                time.sleep(10)
                try:
                    run(passwd, model_path, dev_type, threads, iterations)
                    sync_command = ["sudo", "sync"]
                    subprocess.run(sync_command, input=passwd, universal_newlines=True)
                    drop_caches_command = ["sudo", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
                    subprocess.run(drop_caches_command, input=passwd, universal_newlines=True)
                except Exception as e:
                    logging.exception(f"Exception occurred, error {e}")

def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)

def j_mode(id_mode : int, clock : bool):
    from jtop import jtop
    with jtop() as jetson:
        if jetson.ok():
            jetson.nvpmodel = id_mode
            jetson.jetson_clocks = clock

def jetson_stat():
    from jtop import jtop
    jetson = jtop()
    jetson.start()
    nvp = jetson.nvpmodel
    if jetson.jetson_clocks.status == 'running':
        jetson.close()
        return 1, nvp
    else:
        jetson.close()
        return 0, nvp

if __name__ == '__main__':
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the model', required=True)
    parser.add_argument('--dev_type', help='device type | see in yaml file list', required=True)
    parser.add_argument('--threads', help='num_threads (just for tflite)', default=None)
    parser.add_argument('--iterations', help='how many model runs (not including warm-up)', required=True)
    config = configparser.ConfigParser()
    config.read("._config.ini")
    _passwd = config.get("Credentials", "password", raw=True)
    args = parser.parse_args()
    logging.basicConfig(filename=f'errorlog.log', filemode='w')
    main(_passwd, args.model_path, args.dev_type, (int(args.threads) if isinstance(args.threads, int) else None), int(args.iterations))