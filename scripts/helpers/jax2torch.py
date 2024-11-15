import os

import torch
from dlpack import asdlpack


def convert_pickle_to_pt(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pt"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as f:
                    jax_array = torch.load(f)
                print(type(jax_array), file_path)
                # Convert JAX array to PyTorch tensor
                # torch_tensor = torch.tensor(torch.from_dlpack(asdlpack(jax_array)))

                # Save the tensor as a .pt file
                # pt_file_path = file_path.replace(".pickle", ".pt")
                # torch.save(torch_tensor, pt_file_path)
                # print(f"Converted {file_path} to {pt_file_path}")


if __name__ == "__main__":
    root_directory = "/Users/kcvejoski/Projects/TemporalFoundatioinModels/FIM/scripts/huggingface/datasets/mjp/data"
    convert_pickle_to_pt(root_directory)
