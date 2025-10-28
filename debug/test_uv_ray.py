import os
import sys

import ray

print("Driver Python:", sys.executable)
print("Driver Ray:", ray.__version__)
print("Driver VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))