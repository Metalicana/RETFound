import numpy as np

with np.load("/lustre/fs1/home/yu395012/RETFound/OphthalmicAgent/data2/data2/RNFLT/data_0005.npz") as data:
    for key in data.files:
        array = data[key]
        print(f"Array name: {key}")
        print(f"  Shape:    {array.shape}")
        print(f"  Data Type: {array.dtype}")
        print("-" * 20)
    
    print(f"MD score: {data["md"]}")
    print(f"Glaucoma: {data["glaucoma"]}")
    print(f"Progression: {data["progression"]}")