import os
import subprocess
import threading
import time
import psutil
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

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

def rss_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def get_size(file_path, unit='MB'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'Bytes': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from ['Bytes', 'KB', 'MB', 'GB']")
    else:
        size = file_size / (1024 ** exponents_map[unit])
        return round(size, 2)

def clear_cache(passwd):
    subprocess.run(["echo", passwd, "|", "sudo", "-S", "sync"], shell=True, universal_newlines=True)
    subprocess.run(["echo", passwd, "|", "sudo", "-S", "su", "-c", "echo 3 > /proc/sys/vm/drop_caches"], shell=True, universal_newlines=True)

def evaluate(model_path: str, batch_size, passwd):
    test_set = tfds.load(
            name='oxford_flowers102',
            split='test[:20%]',
            with_info=False,
            as_supervised=True
        )

    model_name = os.path.basename(model_path)
    print("Model Name:", model_name)

    if 'MobileNetV3' in model_name:
        from keras.applications.mobilenet_v3 import preprocess_input
    elif 'EfficientNet' in model_name:
        from keras.applications.efficientnet import preprocess_input
    elif 'DenseNet' in model_name:
        from keras.applications.densenet import preprocess_input

    def resize_image(image, label):
        image = tf.image.resize(image, size=(224, 224))
        image = tf.cast(image, dtype=tf.float32)
        image = preprocess_input(image)
        return image, label

    test_set = test_set.map(map_func=resize_image, num_parallel_calls=tf.data.AUTOTUNE)

    print("Batch Size:", batch_size)
    data = {'Model': [], 'Model Size (MB)': [], 'Dataset':['oxford_flowers102'],
    'Num. of Test Imgs':[len(test_set)], 'Batch Size': [],
    'Accuracy (%)': [], 'Latency (ms)': [], 'CPU Usage (%)': [],
    'Memory Usage (MB)': []}
    data['Batch Size'].append(batch_size)
    data['Model'].append(model_name)
    test_set_batched = test_set.batch(batch_size=batch_size)

    data['Model Size (MB)'].append(get_size(model_path, 'MB'))

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.resize_tensor_input(0, [batch_size, 224, 224, 3])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predicted_labels = []
    true_labels = []

    perf = {'Latency (ms)': [], 'CPU Usage (%)': [],
    'Memory Usage (MB)': []}

    for image_batch, label_batch in test_set_batched:
        for index in range(len(label_batch)):
            true_labels.append(label_batch[index].numpy())

        thread = CPU()
        thread.start()
        time.sleep(2)
        start = time.time()

        batch_images = np.stack(image_batch)  # Stack images to create a batch
        batch_images = batch_images.reshape((5, 224, 224, 3)).astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], batch_images)
        interpreter.invoke()

        # Assuming the output tensor is 1D, adjust accordingly if it's different
        output = interpreter.get_tensor(output_details["index"])
        predictions = np.argmax(output, axis=1)
        predicted_labels.extend(predictions)

        elapsed = (time.time() - start) * 1000
        perf['Latency (ms)'].append(elapsed)
        perf['Memory Usage (MB)'].append(rss_memory())
        if elapsed < 1000:
            time.sleep((2000-elapsed)/1000)
        thread.stop()
        thread.join()
        perf['CPU Usage (%)'].append(float(thread.result[0]))
        # clear cache
        clear_cache(passwd)
    
    data['Latency (ms)'].append(round(sum(perf['Latency (ms)'])/len(perf['Latency (ms)']), 2))
    data['Memory Usage (MB)'].append(round((sum(perf['Memory Usage (MB)'])/len(perf['Memory Usage (MB)']))/1024**2, 2))
    data['CPU Usage (%)'].append(round(sum(perf['CPU Usage (%)'])/len(perf['CPU Usage (%)']), 2))

    accurate_count = np.sum(np.array(predicted_labels) == np.array(true_labels))
    data['Accuracy (%)'].append(round(accurate_count * 100.0 / len(predicted_labels), 2))

    print("*" * 25)

    df = pd.DataFrame(data)
    output_path = 'DLperformance_list.csv'
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False)
    else:
        df.to_csv(output_path)

    print('='*25)

if __name__ == '__main__':
    import argparse
    import configparser

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='Path of the model path', required=True)
    parser.add_argument('--batch_size', help="batch size", default=5)
    config = configparser.ConfigParser()
    config.read("._config.ini")
    _passwd = config.get("Credentials", "password", raw=True)
    args = parser.parse_args()

    st = time.time()
    evaluate(args.model_path, int(args.batch_size), _passwd)
    print(f"DLperformance Collector has measured in {time.time() - st} seconds")
