# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch
from fim import test_data_path
from fim.models import ModelFactory

# from reasoningschema.data import load_tokenizer
# from reasoningschema.data.dataloaders import DataLoaderFactory
# from reasoningschema.models import HSN, AModel, ModelFactory
# from reasoningschema.models.blocks.decoders import Decoder
# from reasoningschema.models.blocks.encoders import EncoderModelA
from fim.utils.helper import GenericConfig, load_yaml


class TestAR:
    @pytest.fixture
    def train_config(self):
        conf_path = test_data_path / "config" / "ar_lstm_vanila.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    def test_init_ar(self, train_config: GenericConfig):
        model = ModelFactory.create(
            name="AR", recurrent_module=train_config.model.recurrent_module.to_dict(), output_head=train_config.model.output_head.to_dict()
        )
        assert model is not None
        assert model.device == torch.device("cpu")
        del model
        model = ModelFactory.create(
            name="AR",
            recurrent_module=train_config.model.recurrent_module.to_dict(),
            output_head=train_config.model.output_head.to_dict(),
            device_map="cuda",
        )
        assert model is not None
        assert model.device == torch.device("cuda:0")
        del model


#     @pytest.mark.skip()
#     def test_model_factory_seq2seq(self):
#         backbone = "MBZUAI/LaMini-T5-61M"
#         tokenizer = load_tokenizer(backbone)
#         assert tokenizer is not None
#         model_params = {"backbone": backbone}
#         model = ModelFactory.create("LLMSeq2Seq", **model_params)
#         assert model is not None

#     def test_model_factory_causal(self):
#         backbone = "meta-llama/Llama-2-7b-chat-hf"
#         tokenizer = load_tokenizer(backbone, add_pad_token=True)
#         num_added_tokens = len(tokenizer.added_tokens_decoder)
#         assert tokenizer is not None
#         model_params = {"backbone": backbone, "device_map": "cuda", "peft": {"method": None}}
#         num_added_tokens = len(tokenizer.added_tokens_decoder)
#         model = ModelFactory.create("LLMCausal", **model_params, pad_token_id=tokenizer.pad_token_id, num_added_tokens=num_added_tokens)
#         assert model is not None
#         # dataloader = DataLoaderFactory.create(
#         #     "commonsense_qa",
#         #     batch_size=1,
#         #     tokenizer=tokenizer,
#         #     max_padding_length=100,
#         #     supervised=True,
#         #     output_fields=["input_ids", "attention_mask", "labels"],
#         #     force_download=True,
#         # )
#         dataloader = DataLoaderFactory.create(
#             "commonsense_qa",
#             batch_size=1,
#             tokenizer=tokenizer,
#             supervised=True,
#             max_padding_length=100,
#             target_type="INSTRUCTION_FINTUNE",
#             force_download=True,
#             output_fields=["input_ids", "attention_mask", "labels"],
#         )
#         minibatch = next(iter(dataloader.train_it))
#         model.train()
#         for k, v in minibatch.items():
#             minibatch[k] = v.to("cuda")
#         losses = model(batch=minibatch)["losses"]
#         assert isinstance(losses, dict)


# # class TestHSNGPT2:
# #     @pytest.fixture
# #     def train_config(self):
# #         conf_path = test_data_path / "config" / "hsn_gpt2_commonsenseqa.yaml"
# #         train_config = load_yaml(conf_path, True)
# #         return train_config

# #     @pytest.fixture
# #     def tokenizer(self, train_config):
# #         tokenizer = load_tokenizer(**train_config.tokenizer.to_dict())
# #         return tokenizer

# #     @pytest.fixture
# #     def model(self, train_config, tokenizer):
# #         self.device_map = train_config.experiment.device_map
# #         num_added_tokens = len(tokenizer.added_tokens_decoder)
# #         model = ModelFactory.create(**train_config.model.to_dict(), pad_token_id=tokenizer.pad_token_id, num_added_tokens=num_added_tokens)

# #         return model.to(self.device_map)

# #     @pytest.fixture
# #     def dataset(self, train_config, tokenizer):
# #         dataloader = DataLoaderFactory.create(**train_config.dataset.to_dict(), tokenizer=tokenizer)
# #         return dataloader.train_it

# #     @pytest.fixture
# #     def schedulers(self, train_config, dataset):
# #         schedulers_config = train_config.trainer.schedulers
# #         max_steps = train_config.trainer.epochs * len(dataset)
# #         return create_schedulers(schedulers_config, max_steps, len(dataset))

# #     def test_init_hsn(self, model):
# #         assert isinstance(model, HSN)
# #         assert isinstance(model.encoder, EncoderModelA)
# #         assert isinstance(model.decoder, Decoder)

# #     def test_forward(self, model: AModel, dataset):
# #         for batch in dataset:
# #             for key in batch.keys():
# #                 batch[key] = batch[key].to(self.device_map)
# #             out = model(batch)
# #             break
# #         assert isinstance(out["losses"], dict)
# #         assert out["histograms"]["paths"].shape == (16, 10, 50)
# #         assert out["logits"].shape == (16, 130, 50432)

# #     def test_forward_schedulers(self, model: AModel, dataset, schedulers: dict):
# #         for batch in dataset:
# #             for key in batch.keys():
# #                 batch[key] = batch[key].to(self.device_map)
# #             out = model(batch, schedulers=schedulers, step=1000)
# #             break
# #         assert isinstance(out["losses"], dict)
# #         assert out["histograms"]["paths"].shape == (16, 10, 50)
# #         assert out["logits"].shape == (16, 130, 50432)

# #     def test_forward_no_labels(self, model: AModel, dataset):
# #         for batch in dataset:
# #             del batch["labels"]
# #             for key in batch.keys():
# #                 batch[key] = batch[key].to(self.device_map)

# #             logits = model(batch)
# #             break

# #         assert logits.shape == (16, 130, 50432)

# #         assert logits.shape == (16, 130, 50432)
