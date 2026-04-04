import json

with open("data/phase1_gpu_architecture.JSON", encoding="utf-8") as f:
    content = f.read()

char = 37219
print(repr(content[char-200:char+200]))