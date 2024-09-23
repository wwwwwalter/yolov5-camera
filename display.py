import os
import pathlib

path="1","4","3"
print(type(path))
print(path)

for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:

    print(p)

