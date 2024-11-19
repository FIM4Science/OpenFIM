from dataclasses import dataclass, field
from typing import List, Optional
from fim import results_path

@dataclass
class FIMSDEConfig:
    
    # saving 
    experiment_name:str = "sde"
    experiment_dir:str = rf"{results_path}"

    # phi_0^t
    temporal_embedding_size: int = 19

    # phi_0^s
    spatial_embedding_size: int = 19
    spatial_embedding_hidden_layers: Optional[int] = field(default_factory=lambda: [25])  # If null, it will just be a dense layer

    # psi_1
    sequence_encoding_tokenizer: int = 5
    sequence_encoding_transformer_hidden_size: int = 50
    sequence_encoding_transformer_heads: int = 1
    sequence_encoding_transformer_layers: int = 1

    # Omega_1
    combining_transformer_hidden_size: int = 50
    combining_transformer_heads: int = 1
    combining_transformer_layers: int = 1

    # phi_1
    trunk_net_size: int = 50
    trunk_net_hidden_layers: Optional[int] = field(default_factory=lambda: [25])

    # optimizer + regularization
    num_epochs:int = 2
    add_delta_x_to_value_encoder: bool = True
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    dropout_rate: float = 0.1

    # loss settings
    diffusion_loss_scale: float = 1.0
    loss_threshold: float = 100.0
    loss_type:str = "rmse" #var, rmse

    log_images_every_n_epochs:int=2
    train_with_normalized_head:bool = True
    clip_grad:bool = True
    clip_max_norm:float = 10.

    # INFERENCE/PIPELINE ------------------------------------------------------------
    dt_pipeline:float = 0.01
    number_of_time_steps_pipeline:int = 128
    evaluate_with_unnormalized_heads:bool = True
    
# Example instantiation
model_config = FIMSDEConfig(
    temporal_embedding_size=256,
    spatial_embedding_size=256,
    spatial_embedding_hidden_layers=None,  # or you can specify a list if needed
    sequence_encoding_transformer_hidden_size=256,
    sequence_encoding_transformer_heads=8,
    sequence_encoding_transformer_layers=4,
    combining_transformer_hidden_size=256,
    combining_transformer_heads=8,
    combining_transformer_layers=4,
    trunk_net_size=576,
    trunk_net_hidden_layers=[256, 256],
    add_delta_x_to_value_encoder=True,
    learning_rate=1.0e-5,
    weight_decay=1.0e-4,
    dropout_rate=0.1,
    diffusion_loss_scale=1.0,
    loss_threshold=100.0
)
