import gdown
import os

tom_id = "1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy"
output = "cp-vton/checkpoints/tom_final.pth"

os.makedirs(os.path.dirname(output), exist_ok=True)

if not os.path.exists(output):
    print(f"Downloading TOM model to {output}...")
    gdown.download(id=tom_id, output=output, quiet=False)
else:
    print(f"TOM model already exists at {output}")
