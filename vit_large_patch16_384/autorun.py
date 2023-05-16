import datetime
import subprocess
import time

# 设置要执行命令的时间点，这里设置为2023年4月19日10点0分0秒
execute_time = datetime.datetime(2023, 4, 18, 13, 10, 0)

# 计算当前时间距离执行时间的时间差，并等待
wait_time = (execute_time - datetime.datetime.now()).total_seconds()
time.sleep(wait_time)

# 执行指定的命令
subprocess.run('python /root/autodl-tmp/kaggle/img2text/train.py', shell=True)

