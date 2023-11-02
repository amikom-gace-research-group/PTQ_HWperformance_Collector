import os
import ast
import yaml
import pandas as pd
import time
import subprocess

def run(memaloc, passwd, model_path, dev_type, iteration, cgroup_name):
    print(f"Physical Memory Limit : {memaloc}Mb")
    model_name = os.path.basename(model_path)
    print("Model : ", model_name)
    os.system(f'echo {passwd} | sudo -S su -c "echo {memaloc}M > /sys/fs/cgroup/memory/{cgroup_name}/memory.limit_in_bytes"')
    template = {'Model':[model_name], 'Memory Allocation (Mb)':[memaloc], 'Model Size (Mb)':[get_size(model_path, 'mb')], 'Warmup-Latency (ms)':[], 'Warmup-CPU Usage (%)':[], 'Warmup-Mem RSS Usage (Mb)':[], 'Warmup-Mem PSS Usage (Mb)':[], 'Warmup-Mem USS Usage (Mb)':[], 'Warmup-Power (mW)':[]}
    time.sleep(10)
    print('Iteration :', iteration)
    cmd = subprocess.check_output(f'echo {passwd} | sudo -S cgexec -g memory:{cgroup_name} python3 dlpref_meter/benchmark.py --model {model_path} --type {dev_type} --iteration {iteration} --passwd {passwd}', shell=True)
    res = cmd.decode('utf-8')
    data = ast.literal_eval(res)
    for idx, j in enumerate(data):
        if 'tflite' in dev_type:
            if idx == 0:
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-CPU Usage (%)'].append(float(j[1]))
                template['Warmup-Mem RSS Usage (Mb)'].append(float(j[2][0]))
                template['Warmup-Mem PSS Usage (Mb)'].append(float(j[2][1]))
                template['Warmup-Mem USS Usage (Mb)'].append(float(j[2][2]))
                template['Warmup-Power (mW)'].append(float(j[3]))
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory PSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory USS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Power (Lat-{idx}) (mW)'] = []
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'CPU Usage (Lat-{idx}) (%)'].append(float(j[1]))
                template[f'Memory RSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][0]))
                template[f'Memory PSS Usage (Lat-{idx}) (Mb)'].append(float(j[2][1]))
                template[f'Memory USS Usage (Lat-{idx}) (Mb)'].append(float(j[2][2]))
                template[f'Power (Lat-{idx}) (mW)'].append(float(j[3]))
        elif 'trt' in dev_type:
            if idx == 0:
                if 'Warmup-GPU Usage (%)' not in template:
                    template['Warmup-GPU Usage (%)'] = []
                    template['Warmup-J_Clock'] = []
                template['Warmup-Latency (ms)'].append(float(j[0]))
                template['Warmup-GPU Usage (%)'].append(float(j[1]))
                template['Warmup-CPU Usage (%)'].append(float(j[2]))
                template['Warmup-Mem RSS Usage (Mb)'].append(float(j[3][0]))
                template['Warmup-Mem PSS Usage (Mb)'].append(float(j[3][1]))
                template['Warmup-Mem USS Usage (Mb)'].append(float(j[3][2]))
                template['Warmup-Power (mW)'].append(float(j[4]))
                template['Warmup-J_Clock'].append(jclock_stat())
            else:
                if f'Latency {idx} (ms)' not in template:
                    template[f'Latency {idx} (ms)'] = []
                    template[f'CPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Memory RSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory PSS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'Memory USS Usage (Lat-{idx}) (Mb)'] = []
                    template[f'GPU Usage (Lat-{idx}) (%)'] = []
                    template[f'Power (Lat-{idx}) (mW)'] = []
                    template[f'Lat({idx})-J_Clock'] = []
                template[f'Latency {idx} (ms)'].append(float(j[0]))
                template[f'GPU Usage (Lat-{idx}) (%)'].append(float(j[1]))
                template[f'CPU Usage (Lat-{idx}) (%)'].append(float(j[2]))
                template[f'Memory RSS Usage (Lat-{idx}) (Mb)'].append(float(j[3][0]))
                template[f'Memory PSS Usage (Lat-{idx}) (Mb)'].append(float(j[3][1]))
                template[f'Memory USS Usage (Lat-{idx}) (Mb)'].append(float(j[3][2]))
                template[f'Power (Lat-{idx}) (mW)'].append(float(j[4]))
                template[f'Lat({idx})-J_Clock'].append(jclock_stat())

    df = pd.DataFrame(template)
    output_path=f'latency_{dev_type}.csv'
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False)
    else: df.to_csv(output_path)
    print('='*25)

def main(passwd, model_path, dev_type, iteration, cgroup_name):
    with open('scenario.yml', 'r') as yml:
        scenarios = yaml.safe_load(yml)
    for g in range(scenarios[dev_type]['start'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in range(5):
            try:
                run(g, passwd, model_path, dev_type, iteration, cgroup_name)
            except Exception as e:
                print(f"Memory Allocation {g}M Cannot Run, Error {e}")
                continue
    for k in reversed(range(scenarios[dev_type]['start'], scenarios[dev_type]['stop'], scenarios[dev_type]['stage'])):
        for _ in range(5):
            try:
                run(k, passwd, model_path, dev_type, iteration, cgroup_name)
            except Exception as e:
                print(f"Memory Allocation {k}M Cannot Run, Error {e}")
                continue
    for l in range(scenarios[dev_type]['start']+scenarios[dev_type]['stage'], scenarios[dev_type]['stop']+scenarios[dev_type]['stage'], scenarios[dev_type]['stage']):
        for _ in range(5):
            try:
                run(l, passwd, model_path, dev_type, iteration, cgroup_name)
            except Exception as e:
                print(f"Memory Allocation {l}M Cannot Run, Error {e}")
                continue

def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)

def jclock_stat():
    from jtop import jtop
    jetson = jtop()
    jetson.start()
    if jetson.jetson_clocks.status == 'running':
        return 1
    else:
        return 0
    jetson.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the model', required=True)
    parser.add_argument('--dev_type', help='device type | see in yaml file list', required=True)
    parser.add_argument('--iteration', help='how many model runs (not including warm-up)', required=True)
    parser.add_argument('--cgroup_name', help='cgroup name named in cgroup settings', required=True)
    args = parser.parse_args()
    passwd = input("Password : ")

    main(passwd, args.model_path, args.dev_type, int(args.iteration), args.cgroup_name)
    os.system(f"echo {passwd} | sudo -S sync; sudo -S su -c 'echo 3 > /proc/sys/vm/drop_caches'")
