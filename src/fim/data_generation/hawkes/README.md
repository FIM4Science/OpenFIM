## Generating data

These are the steps to generate data

1. Fill the configuration file `configs/data_generation/hawkes/data_generation.yaml`

2. Run this command:
   ```bash
   python3 scripts/generate_dataset.py -c configs/data_generation/hawkes/data_generation.yaml
   ```

3. The synthetic data will be stored in `data/synthetic_data/hawkes`
