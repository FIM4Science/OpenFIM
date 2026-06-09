
# VDP and FHN data

If you want to run experiments on the Vanderpol or FHN data, you must first generate it using

```bash
uv run python experiments/data_gen_vdp_fhn_gpode.py
```

To run MoCap experiments, first generate the data using

```bash
uv run python experiments/data_gen_mocap.py
```


# ODEFormer

To run experiments using ODEFormer, install it by

```bash
uv pip install git+https://github.com/osorensen/odeformer.git
````

Do not do ```uv pip install odeformer```, the above is a fork that fixed some issues.
