from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = True
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # layout for each token you need to optimize
    bbox: List[int] = None
    # The token you need to optime is NOUN or Modifier
    part_of_speech : List[int] = None
    # Which random seeds to use when generating
    seeds: int = 42
    # The number of images to generate for each prompt
    per_num : int = 10
    # Need cross attention map visualization
    attention_vis : bool = False
    # Path to save all outputs to
    output_path: Path = Path('./output')
    # Path to dataset
    data_path :str = "./T2I-CompBench_dataset/"
    # Path to prompt layout which generate by GPT
    layout_path : str = "./layout_generation/"
    # Path to Stable diffusion model
    model_path : str = None
    # Path to save attention visualization image
    attention_vis_file : str = "./visualization_outputs/"
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply optimize
    max_iter_to_alter: int = 30
    soft_mask_rate: float = 0.1
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    # Dict{key:[PCA threshold, RCA threshold]}
    thresholds: Dict[int, List[float]] = field(default_factory=lambda: {0:[[0.15],[0.1]], 5:[[0.15],[0.4]], 10:[[0.6],[0.6]], 20:[[0.9],[0.7]], 30:[[0.95],[0.8]]})
    # Which time steps to save the cross attention map if you need
    att_vis_timestep: List = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)



