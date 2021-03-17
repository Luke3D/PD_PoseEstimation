import os
from pathlib import Path
import numpy as np

basepath = Path('/Users/luca/Box/CISPD_Videos_proc/Edited/')
fullfiles = []
# dirs = os.listdir(basepath) #all subjs
dirs = [1004, 1009, 1019, 1023, 1039, 1043, 1044, 1047, 1049, 1050, 1052, 1054, 1055, 1056]
for s in dirs:
    s = Path(str(s))
    fullfiles.append([basepath/s/Path(f) for f in os.listdir(basepath/s) if 'RamR_1' in f or 'RamR_2' in f])

print(fullfiles)