from datasets import load_dataset


dataset = load_dataset(
    "/home/cvejoski/Projects/FoundationModels/FIM/scripts/huggingface/datasets/mjp",
    trust_remote_code=True,
    download_mode="force_redownload",
)

print(dataset)
