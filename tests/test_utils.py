# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


# TRAIN_CONF = test_data_path / "config" / "gpt2_commonsense_qa.yaml"


# def test_create_optimizer():
#     config = load_yaml(TRAIN_CONF, True)
#     print(config.optimizers)
#     model = ModelFactory.create(**config.model.to_dict())
#     optimizers = create_optimizers(model, config.optimizers)
#     print(optimizers)
#     assert optimizers is not None
#     assert isinstance(optimizers["optimizer_d"]["opt"], torch.optim.AdamW)
