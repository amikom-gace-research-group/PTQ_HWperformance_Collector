import os
import ast
import yaml
import pandas as pd
import numpy as np
import time
import psutil
import logging
import subprocess
from platform import uname
from dlperf_meter.benchmark import check_ina219

def run(memaloc : int, passwd : str, model_path : str, dev_type : str, threads, iterations : int, cgroup_name : str):
    print(f"Physical Memory Limit : {memaloc}Mb")
    model_name = os.path.basename(model_path)
    print("Model : ", model_name)
    memory_limit_command = [
    "sudo", "su", "-c", f"echo {memaloc}M > /sys/fs/cgroup/memory/{cgroup_name}/memory.limit_in_bytes"
    ]
    subprocess.run(memory_limit_command, input=passwd, universal_newlines=True)
    if 'gpu' in dev_type:
        template = {'Model':[model_name], 'Memory Allocation (MB)':[memaloc], 'Model Size (MB)':[get_size(model_path, 'mb')], 'J_Clock':[jetson_stat()[0]], 'J_NVP':[jetson_stat()[1]], 'CPU Cores':[psutil.cpu_count()], 'Warmup-CPU Freq (MHz)':[], 'Warmup-GPU Freq (MHz)':[], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (MB)':[], 'Warmup-Mem Swap Usage (MB)':[], 'Warmup-Mem GPU Usage (MB)':[], 'Warmup-Power (mW)':[], 'Warmup-Power CPU (mW)':[], 'Warmup-Power GPU (mW)':[], 'Warmup-GPU Usage (%)':[]}
    else:
        template = {'Model':[model_name], 'Memory Allocation (MB)':[memaloc], 'Model Size (MB)':[get_size(model_path, 'mb')], 'Num Threads':[threads], 'CPU Cores':[psutil.cpu_count()], 'Warmup-CPU Freq (MHz)':[], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (MB)':[], 'Warmup-Mem Swap Usage (MB)':[]}
    time.sleep(10)
    print('iterations :', iterations)
    benchmark_command = [
    "sudo", "cgexec", "-g", f"memory:{cgroup_name}",
    "python3", "dlperf_meter/benchmark.py",
    "--model", model_path,
    "--type", dev_type,
    "--threads", threads,
    "--iterations", iterations
    ]
    cmd = subprocess.run(benchmark_command, input=passwd, capture_output=True, universal_newlines=True).stdout
    res = cmd.decode('utf-8')
    data = ast.literal_eval(res)
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
                        if 'Warmup-Power CPU (mW)' not in template:
                            template['Warmup-Power CPU (mW)'] = []
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
    output_path=f'dlperf_{dev_type}_{uname().release}.csv'
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False)
    else: df.to_csv(output_path)
    print('='*25)

def main(passwd : str, model_path : str, dev_type: str, threads, iterations : int, cgroup_name : str):
    with open('scenario.yml', 'r') as yml:
        scenarios = yaml.safe_load(yml)
    for g in np.arange(scenarios[dev_type]['start'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in np.arange(10):
            try:
                run(g, passwd, model_path, dev_type, threads, iterations, cgroup_name)
                sync_command = ["sudo", "sync"]
                subprocess.run(sync_command, input=passwd, universal_newlines=True)
                drop_caches_command = ["sudo", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
                subprocess.run(drop_caches_command, input=passwd, universal_newlines=True)
            except Exception as e:
                logging.exception(f"Exception occurred, error {e}")
    for k in reversed(np.arange(scenarios[dev_type]['start'], scenarios[dev_type]['stop'], scenarios[dev_type]['stage'])):
        for _ in np.arange(10):
            try:
                run(k, passwd, model_path, dev_type, threads, iterations, cgroup_name)
                sync_command = ["sudo", "sync"]
                subprocess.run(sync_command, input=passwd, universal_newlines=True)
                drop_caches_command = ["sudo", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
                subprocess.run(drop_caches_command, input=passwd, universal_newlines=True)
            except Exception as e:
                logging.exception(f"Exception occurred, error {e}")
    for l in np.arange(scenarios[dev_type]['start']+scenarios[dev_type]['stage'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in np.arange(10):
            try:
                run(l, passwd, model_path, dev_type, threads, iterations, cgroup_name)
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
    parser.add_argument('--cgroup_name', help='cgroup name named in cgroup settings', required=True)
    config = configparser.ConfigParser()
    config.read("._config.ini")
    _passwd = config.get("Credentials", "password")
    args = parser.parse_args()
    logging.basicConfig(filename=f'errorlog.log', filemode='w')
    main(_passwd, args.model_path, args.dev_type, (int(args.threads) if isinstance(args.threads, int) else None), int(args.iterations), args.cgroup_name)
