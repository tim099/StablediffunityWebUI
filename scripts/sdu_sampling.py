'''
this is a customize version of k-diffusion/sampling.py
in order to add some customize action to sample steps
https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
'''

import os
import sys
import torch
import inspect
from tqdm.auto import trange, tqdm
from scripts.global_scripts.sdu_sample_data import SampleData
from scripts.global_scripts.sdu_globals import global_setting
from modules.sd_samplers_kdiffusion import samplers_k_diffusion

from k_diffusion.sampling import default_noise_sampler
from k_diffusion.sampling import get_ancestral_step
from k_diffusion.sampling import to_d
from k_diffusion.sampling import linear_multistep_coeff


def sample_start(model, sampler, x, sigmas, extra_args=None, callback=None, disable=None)-> SampleData:
    sampler_name = "Unknown"

    for samplers_setting in samplers_k_diffusion:
        if samplers_setting[1] == sampler:
            sampler_name = samplers_setting[0]
    print(f"SDU sample_start sampler:{sampler}({sampler_name})\n",flush=True)

    print(f"sigmas:" + ", ".join(f'{x.item():.3f}' for x in sigmas) + "\n",flush=True)
    global_setting.sample_start()
    sample_data = SampleData()
    sample_data.x = x

    return sample_data
def sample_end(sample_data:SampleData):
    #sys.stdout.flush()
    print("SDU sample_end\n",flush=True)
    print("skip_steps:" + ", ".join(f'{str(step)}' for step in sample_data.skip_steps)+"\n",flush=True)

@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack

    for i in trange(len(sigmas) - 1, disable=disable):

        #SDU Hijack Start
        sample_data.step = i
        global_setting.trigger(sample_data)
        if(global_setting.skip_sample(sample_data)):
            continue
        #SDU Hijack End

        denoised = model(sample_data.x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': sample_data.x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            sample_data.x = (sigma_fn(t_next) / sigma_fn(t)) * sample_data.x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            sample_data.x = (sigma_fn(t_next) / sigma_fn(t)) * sample_data.x - (-h).expm1() * denoised_d
        old_denoised = denoised

    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    
    sample_data = sample_start(model,"sample_euler(Euler)", x, sigmas, extra_args, callback, disable)#SDU Hijack

    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        #SDU Hijack Start
        sample_data.step = i
        global_setting.trigger(sample_data)
        if(global_setting.skip_sample(sample_data)):
            continue
        #SDU Hijack End
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(sample_data.x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            sample_data.x = sample_data.x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(sample_data.x, sigma_hat * s_in, **extra_args)
        d = to_d(sample_data.x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': sample_data.x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        sample_data.x = sample_data.x + d * dt

    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack

    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(sample_data.x) if noise_sampler is None else noise_sampler
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        #SDU Hijack Start
        sample_data.step = i
        global_setting.trigger(sample_data)
        if(global_setting.skip_sample(sample_data)):
            continue
        #SDU Hijack End

        denoised = model(sample_data.x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': sample_data.x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(sample_data.x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        sample_data.x = sample_data.x + d * dt
        if sigmas[i + 1] > 0:
            sample_data.x = sample_data.x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

    sample_end(sample_data)#SDU Hijack
    return sample_data.x

@torch.no_grad()
def sample_lms(model, x, sigmas, extra_args=None, callback=None, disable=None, order=4):
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    extra_args = {} if extra_args is None else extra_args
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=disable):
        #SDU Hijack Start
        sample_data.step = i
        global_setting.trigger(sample_data)
        if(global_setting.skip_sample(sample_data)):
            continue
        #SDU Hijack End

        denoised = model(sample_data.x, sigmas[i] * s_in, **extra_args)
        d = to_d(sample_data.x, sigmas[i], denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({'x': sample_data.x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
        sample_data.x = sample_data.x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

    sample_end(sample_data)#SDU Hijack
    return sample_data.x

@torch.no_grad()
def sample_heun(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    extra_args = {} if extra_args is None else extra_args
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        #SDU Hijack Start
        sample_data.step = i
        global_setting.trigger(sample_data)
        if(global_setting.skip_sample(sample_data)):
            continue
        #SDU Hijack End

        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(sample_data.x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            sample_data.x = sample_data.x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(sample_data.x, sigma_hat * s_in, **extra_args)
        d = to_d(sample_data.x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': sample_data.x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            sample_data.x = sample_data.x + d * dt
        else:
            # Heun's method
            x_2 = sample_data.x + d * dt
            denoised_2 = model(x_2, sigmas[i + 1] * s_in, **extra_args)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            sample_data.x = sample_data.x + d_prime * dt

    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_dpm_2(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

