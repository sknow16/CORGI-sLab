import argparse
import inspect

import guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

NUM_CLASSES = 1000

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

def create_gaussian_diffusion(
    *,
    # 以下に引数
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False, # 損失でKL使うか
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False, # 
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type=gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type=gd.LossType.RESCALED_MSE
    else:
        loss_type=gd.LossType.MSE
        
    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas = betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

