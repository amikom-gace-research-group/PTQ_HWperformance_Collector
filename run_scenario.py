import os
import ast
import yaml
import pandas as pd
import numpy as np
import time
import logging
import subprocess
from platform import uname

def run(memaloc, passwd, model_path, dev_type, threads, iterations, cgroup_name):
    print(f"Physical Memory Limit : {memaloc}Mb")
    model_name = os.path.basename(model_path)
    print("Model : ", model_name)
    os.system(f'echo {passwd} | sudo -S su -c "echo {memaloc}M > /sys/fs/cgroup/memory/{cgroup_name}/memory.limit_in_bytes"')
    if 'gpu' in dev_type:
        template = {'Model':[model_name], 'Memory Allocation (Mb)':[memaloc], 'Model Size (Mb)':[get_size(model_path, 'mb')], 'J_Clock':[jetson_stat()[0]], 'J_NVP':[jetson_stat()[1]], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (Mb)':[], 'Warmup-Mem PSS Usage (Mb)':[], 'Warmup-Mem USS Usage (Mb)':[], 'Warmup-Power (mW)':[]}
    else:
        template = {'Model':[model_name], 'Memory Allocation (Mb)':[memaloc], 'Model Size (Mb)':[get_size(model_path, 'mb')], 'Num Threads':[threads], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (Mb)':[], 'Warmup-Mem PSS Usage (Mb)':[], 'Warmup-Mem USS Usage (Mb)':[]}
    time.sleep(10)
    print('iterations :', iterations)
    cmd = subprocess.check_output(f'echo {passwd} | sudo -S cgexec -g memory:{cgroup_name} python3 dlperf_meter/benchmark.py --model {model_path} --type {dev_type} --threads {threads} --iterations {iterations} --passwd {passwd}', shell=True)
    res = cmd.decode('utf-8')
    data = ast.literal_eval(res)
    for idx, j in enumerate(data):
        if 'cpu' in dev_type:
            if idx == 0:
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-CPU Usage (%)'].append(float(j[1]))
                template['Warmup-Mem RSS Usage (Mb)'].append(float(j[2][0]))
                template['Warmup-Mem PSS Usage (Mb)'].append(float(j[2][1]))
                template['Warmup-Mem USS Usage (Mb)'].append(float(j[2][2]))
                if 'tegra' in uname().release:
                    if 'Warmup-Power (mW)' not in template:
                        template['Warmup-Power (mW)'] = []
                    template['Warmup-Power (mW)'].append(float(j[3]))
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory PSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory USS Usage (Lat-{idx}) (Mb)'] = []
                    if 'tegra' in uname().release:
                        if f'Power (Lat-{idx}) (mW)' not in template:
                            template[f'Power (Lat-{idx}) (mW)'] = []
                        template[f'Power (Lat-{idx}) (mW)'].append(float(j[3]))
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'CPU Usage (Lat-{idx}) (%)'].append(float(j[1]))
                template[f'Memory RSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][0]))
                template[f'Memory PSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][1]))
                template[f'Memory USS Usage (Lat-{idx}) (Mb)'].append(float(j[2][2]))
        elif 'gpu' in dev_type:
            if idx == 0:
                if 'Warmup-GPU Usage (%)' not in template:
                    template['Warmup-GPU Usage (%)'] = []
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-GPU Usage (%)'].append(float(j[3]))
                template['Warmup-CPU Usage (%)'].append(float(j[1]))
                template['Warmup-Mem RSS Usage (Mb)'].append(float(j[2][0]))
                template['Warmup-Mem PSS Usage (Mb)'].append(float(j[2][1]))
                template['Warmup-Mem USS Usage (Mb)'].append(float(j[2][2]))
                template['Warmup-Power (mW)'].append(float(j[4]))
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory PSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory USS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'GPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Power (Lat-{idx}) (mW)'] = []
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'GPU Usage (Lat-{idx}) (%)'].append(float(j[3]))
                template[f'CPU Usage (Lat-{idx}) (%)'].append(float(j[1]))
                template[f'Memory RSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][0]))
                template[f'Memory PSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][1]))
                template[f'Memory USS Usage (Lat-{idx}) (Mb)'].append(float(j[2][2]))
                template[f'Power (Lat-{idx}) (mW)'].append(float(j[4]))

    df = pd.DataFrame(template)
    output_path=f'dlperf_{dev_type}.csv'
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False)
    else: df.to_csv(output_path)
    print('='*25)

def main(passwd, model_path, dev_type, threads, iterations, cgroup_name):
    with open('scenario.yml', 'r') as yml:
        scenarios = yaml.safe_load(yml)
    for g in np.arange(scenarios[dev_type]['start'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in np.arange(5):
            try:
                run(g, passwd, model_path, dev_type, threads, iterations, cgroup_name)
                os.system(f"echo {args.passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")
            except Exception as e:
                logging.exception(f"Exception occurred, error {e}")
    for k in reversed(np.arange(scenarios[dev_type]['start'], scenarios[dev_type]['stop'], scenarios[dev_type]['stage'])):
        for _ in np.arange(5):
            try:
                run(k, passwd, model_path, dev_type, threads, iterations, cgroup_name)
                os.system(f"echo {args.passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")
            except Exception as e:
                logging.exception(f"Exception occurred, error {e}")
    for l in np.arange(scenarios[dev_type]['start']+scenarios[dev_type]['stage'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in np.arange(5):
            try:
                run(l, passwd, model_path, dev_type, threads, iterations, cgroup_name)
                os.system(f"echo {args.passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the model', required=True)
    parser.add_argument('--dev_type', help='device type | see in yaml file list', required=True)
    parser.add_argument('--threads', help='num_threads (just for tflite)', default=None)
    parser.add_argument('--iterations', help='how many model runs (not including warm-up)', required=True)
    parser.add_argument('--cgroup_name', help='cgroup name named in cgroup settings', required=True)
    parser.add_argument('--passwd', help='enter the system password to clear the cache', required=True)
    args = parser.parse_args()
    logging.basicConfig(filename=f'run_scenario_{args.dev_type}.log', filemode='w')
    main(args.passwd, args.model_path, args.dev_type, (int(args.threads) if isinstance(args.threads, int) else None), int(args.iterations), args.cgroup_name)
