import os
import subprocess 

Myout = subprocess.Popen(['dir'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
stdout, stderr = Myout.communicate()
print(stdout)
print(stderr)
