"""Read-only GPU/disk monitoring; never touches the four training processes."""
import csv,json,shutil,subprocess,time
from pathlib import Path

root=Path('results/tables/stage6p1')
with (root/'gpu_resource_samples.csv').open('x',newline='') as stream:
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['timestamp','memory_used_MiB','GPU_utilization_percent','power_W','temperature_C','free_disk_bytes'])
    while True:
        result=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()
        writer.writerow([v.strip() for v in result.split(',')]+[shutil.disk_usage('.').free]);stream.flush()
        if (root/'scheduler_completion.json').exists():break
        time.sleep(20)
