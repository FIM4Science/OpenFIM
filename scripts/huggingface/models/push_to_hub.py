from mjp.configuration_mjp import FIMMJPConfig
from mjp.modeling_mjp import FIMMJP


FIMMJPConfig.register_for_auto_class()
FIMMJP.register_for_auto_class("AutoModel")
n_states = 6
use_adjacency_matrix = False
transformer_block = {
    "name": "fim.models.blocks.TransformerBlock",
    "in_features": 64,
    "ff_dim": 256,
    "dropout": 0.1,
    "attention_head": {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True},
    "activation": {"name": "torch.nn.ReLU"},
    "normalization": {"name": "torch.nn.LayerNorm", "normalized_shape": 64},
}
pos_encodings = {"name": "fim.models.blocks.SineTimeEncoding", "out_features": 64}
timeseries_encoder = {
    "name": "fim.models.blocks.base.TransformerEncoder",
    "num_layers": 4,
    "embed_dim": 64,
    "transformer_block": transformer_block,
}
path_attn = {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True}
intensity_matrix_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
initial_distribution_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
mjp_config = FIMMJPConfig(
    n_states, use_adjacency_matrix, timeseries_encoder, pos_encodings, path_attn, intensity_matrix_decoder, initial_distribution_decoder
)
# mjp = FIMMJP(mjp_config)
mjp = FIMMJP.load_model(
    "/home/cvejoski/Projects/FoundationModels/FIM/results/FIM_MJP_Homogeneous_no_annealing_rnn_256_path_attention_one_head_model_dim_var_path_same-experiment-seed-0_11-06-1535/checkpoints/best-model"
)


mjp.push_to_hub("cvejoski/FIMMJP")

# from transformers import AutoModel

# from fim.models import FIMMJP


# mjp_trans = AutoModel.from_pretrained("cvejoski/FIMMJP", trust_remote_code=True)
# mjp = FIMMJP.from_pretrained("cvejoski/FIMMJP", trust_remote_code=True)
# print(mjp_trans)
