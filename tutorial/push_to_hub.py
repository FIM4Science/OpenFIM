from datasets import Dataset, DatasetDict, load_dataset
import json 

with open("tutorial/double_well_duffing_original_trajectories.json", "r") as f:
    data = json.load(f)

# If your JSON is a list of dictionaries

data_duffing={"observations": data[0]["observations"], "locations": data[0]["locations"], "initial_states": data[0]["initial_states"]}
data_dw_constant={"observations": data[1]["observations"], "locations": data[1]["locations"], "initial_states": data[1]["initial_states"]}
data_dw={"observations": data[2]["observations"], "locations": data[2]["locations"], "initial_states": data[2]["initial_states"]}
dataset_duffing = Dataset.from_dict(data_duffing)
dataset_dw_constant = Dataset.from_dict(data_dw_constant)
dataset_dw = Dataset.from_dict(data_dw)

# Upload to the organization
dataset_duffing.push_to_hub("FIM4Science/sde-tutorial-duffing", private=False)
dataset_dw_constant.push_to_hub("FIM4Science/sde-tutorial-double_well_constant_diffusion", private=False)
dataset_dw.push_to_hub("FIM4Science/sde-tutorial-double_well", private=False)
