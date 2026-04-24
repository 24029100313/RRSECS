import os
from datetime import datetime


def compute_gpu_hours(start, end, num_gpus):
    """
    start / end: type-str
    """
    start_time = datetime.strptime(start, "%H:%M:%S")
    end_time = datetime.strptime(end, "%H:%M:%S")

    if end_time < start_time:
        end_time = end_time.replace(day=end_time.day + 1)

    time_interval = end_time - start_time
    seconds = time_interval.total_seconds()
    hours = seconds / 3600

    gpu_hours = round(hours * num_gpus, 2)

    print(gpu_hours)


file_path = '/root/data4/CVPR2024/U-MCL-2/exp_ablation/vaihingen/UAMCL/r50/1_2_UAMCL_P64_S64_0.5Denoising_0.5Reason-GW0.968/20231115_115954.log'
infos = open(file_path, 'r').readlines()
need = []
for info in infos:
    if info.startswith('['):
        need.append(info.strip())

start_time = need[0].split(' ')[1].split(',')[0]
end_time = need[-1].split(' ')[1].split(',')[0]
compute_gpu_hours(start_time, end_time, 4)
