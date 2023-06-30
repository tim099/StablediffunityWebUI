'''
this is a customize version of stable-diffusion-webui processing.py
in order to add some customize action to process_images_inner
https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/processing.py
'''

import os
import sys

import torch
import modules.shared as shared
import modules.sd_hijack
import numpy as np
import modules.images as images

from PIL import Image, ImageOps
from os import device_encoding

from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.processing import get_fixed_seed,create_infotext,decode_first_stage,apply_overlay,apply_color_correction
from modules.shared import opts, cmd_opts, state
from modules import devices, prompt_parser, masking, sd_samplers, lowvram, generation_parameters_copypaste, extra_networks, sd_vae_approx, scripts, sd_samplers_common, sd_unet
from modules.sd_hijack import model_hijack
import modules.paths as paths

def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    print("SDU hijack process_images_inner")
    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    modules.sd_hijack.model_hijack.apply_circular(p.tiling)
    modules.sd_hijack.model_hijack.clear_comments()

    comments = {}

    p.setup_prompts()

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, comments, iteration, position_in_batch)

    if os.path.exists(cmd_opts.embeddings_dir) and not p.do_not_reload_embeddings:
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    if p.scripts is not None:
        p.scripts.process(p)

    infotexts = []
    output_images = []

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)

            # for OSX, loading the model during sampling changes the generated picture, so it is loaded here
            if shared.opts.live_previews_enable and opts.show_progress_type == "Approx NN":
                sd_vae_approx.model()

            sd_unet.apply_unet()

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            print(f"SDU n:{str(n)}")
            p.iteration = n

            if state.skipped:
                state.skipped = False

            if state.interrupted:
                break

            p.prompts = p.all_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n + 1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n + 1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if p.scripts is not None:
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            if len(p.prompts) == 0:
                break

            p.parse_extra_network_prompts()

            if not p.disable_extra_networks:
                with devices.autocast():
                    extra_networks.activate(p, p.extra_network_data)

            if p.scripts is not None:
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            # params.txt should be saved after scripts.process_batch, since the
            # infotext could be modified by that callback
            # Example: a wildcard processed by process_batch sets an extra model
            # strength, which is saved as "Model Strength: 1.0" in the infotext
            if n == 0:
                with open(os.path.join(paths.data_path, "params.txt"), "w", encoding="utf8") as file:
                    processed = Processed(p, [], p.seed, "")
                    file.write(processed.infotext(p, 0))

            p.setup_conds()

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                shared.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.without_autocast() if devices.unet_needs_upcast else devices.autocast():
                samples_ddim = p.sample(conditioning=p.c, unconditional_conditioning=p.uc, seeds=p.seeds, subseeds=p.subseeds, subseed_strength=p.subseed_strength, prompts=p.prompts)

            x_samples_ddim = [decode_first_stage(p.sd_model, samples_ddim[i:i+1].to(dtype=devices.dtype_vae))[0].cpu() for i in range(samples_ddim.size(0))]
            for x in x_samples_ddim:
                devices.test_for_nans(x, "vae")

            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            if lowvram.is_enabled(shared.sd_model):
                lowvram.send_everything_to_cpu()

            devices.torch_gc()

            if p.scripts is not None:
                p.scripts.postprocess_batch(p, x_samples_ddim, batch_number=n)

            for i, x_sample in enumerate(x_samples_ddim):
                p.batch_index = i

                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restoration.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.scripts is not None:
                    pp = scripts.PostprocessImageArgs(image)
                    p.scripts.postprocess_image(p, pp)
                    image = pp.image

                if p.color_corrections is not None and i < len(p.color_corrections):
                    if opts.save and not p.do_not_save_samples and opts.save_images_before_color_correction:
                        image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                        images.save_image(image_without_cc, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-before-color-correction")
                    image = apply_color_correction(p.color_corrections[i], image)

                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

                if opts.samples_save and not p.do_not_save_samples:
                    images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                if opts.enable_pnginfo:
                    image.info["parameters"] = text
                output_images.append(image)

                if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([opts.save_mask, opts.save_mask_composite, opts.return_mask, opts.return_mask_composite]):
                    image_mask = p.mask_for_overlay.convert('RGB')
                    image_mask_composite = Image.composite(image.convert('RGBA').convert('RGBa'), Image.new('RGBa', image.size), images.resize_image(2, p.mask_for_overlay, image.width, image.height).convert('L')).convert('RGBA')

                    if opts.save_mask:
                        images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask")

                    if opts.save_mask_composite:
                        images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], opts.samples_format, info=infotext(n, i), p=p, suffix="-mask-composite")

                    if opts.return_mask:
                        output_images.append(image_mask)

                    if opts.return_mask_composite:
                        output_images.append(image_mask_composite)

            del x_samples_ddim

            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(output_images) < 2 and opts.grid_only_if_multiple
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(output_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                output_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(grid, p.outpath_grids, "grid", p.all_seeds[0], p.all_prompts[0], opts.grid_format, info=infotext(), short_filename=not opts.grid_extended_filename, p=p, grid=True)

    if not p.disable_extra_networks and p.extra_network_data:
        extra_networks.deactivate(p, p.extra_network_data)

    devices.torch_gc()

    res = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotext(),
        comments="".join(f"{comment}\n" for comment in comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )

    if p.scripts is not None:
        p.scripts.postprocess(p, res)

    return res
