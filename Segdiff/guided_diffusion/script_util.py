import argparse
import inspect

import guided_diffusion.gaussian_diffusion as gd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

from guided_diffusion.unet import UNetModel
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

def create_model(
    image_size,
    in_channels,
    num_channels,
    out_channels,
    cond_img_ch,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    rrdb_blocks,
    deeper_net
):
    if image_size == 256:
        if deeper_net:
            channel_mult = (1, 1, 1, 2, 2, 4, 4)
        else:
            channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(out_channels if not learn_sigma else out_channels*2),
        cond_img_ch=cond_img_ch,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        rrdb_blocks=rrdb_blocks
    )

def create_model_and_diffusion(
    image_size,
    in_channels,
    num_channels,
    out_channels,
    cond_img_ch,
    num_res_blocks=2,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16,8",
    num_heads=4,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0,
    rrdb_blocks=10,
    deeper_net=False,
    
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
): 
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        out_channels,
        cond_img_ch,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        rrdb_blocks,
        deeper_net
    )
    diffusion = create_gaussian_diffusion(
        steps=steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion