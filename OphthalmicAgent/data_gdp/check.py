import numpy as np

with np.load("BScan/data_0005.npz") as data:
    # 1. Print info for all arrays inside the file
    for key in data.files:
        array = data[key]
        print(f"Array name: {key}")
        print(f"  Shape:    {array.shape}")
        print(f"  Data Type: {array.dtype}")
        print("-" * 20)
    
#    # 2. Access specific keys (Note the single quotes inside the double-quoted f-strings)
#    print(f"MD score: {data['md']}")
#    print(f"Glaucoma: {data['glaucoma']}")
#    print(f"Progression: {data['progression']}")