import pathlib

cpkt = pathlib.Path("/home/wsw/210.ckpt")


print(f"Checkpoint path: {cpkt}")

if cpkt.exists():
    print("Path exists")
else:
    print("Path does not exist")

if cpkt.is_file():
    print("Path is a file")
else:
    print("Path is not a file")

try:
    with open(cpkt, 'r') as f:
        print("File can be opened")
except Exception as e:
    print(f"Error opening file: {e}")

import os
print(os.path.isfile(str(cpkt)))