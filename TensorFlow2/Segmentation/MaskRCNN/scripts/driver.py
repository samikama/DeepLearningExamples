import os
import subprocess


evaluate = subprocess.Popen(['./session_eval.sh'])

import time
while True:
    if evaluate.poll() == None:
        time.sleep(10)
        print("still running..")
    else:
        
        print("done")
        import sys
        sys.exit()
