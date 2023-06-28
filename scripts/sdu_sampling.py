'''
this is a customize version of k-diffusion/sampling.py
in order to add some customize action to sample steps
https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
'''

import os
import sys
import torch
import inspect
import math
from torch import nn
from tqdm.auto import trange, tqdm
from scripts.global_scripts.sdu_sample_data import SampleData
from scripts.global_scripts.sdu_globals import global_setting
from modules.sd_samplers_kdiffusion import samplers_k_diffusion

from k_diffusion.sampling import default_noise_sampler
from k_diffusion.sampling import get_ancestral_step
from k_diffusion.sampling import to_d
from k_diffusion.sampling import linear_multistep_coeff
from k_diffusion.sampling import BrownianTreeNoiseSampler
from k_diffusion.sampling import PIDStepSizeController
#from k_diffusion.sampling import DPMSolver
def sample_start(model, sampler, x, sigmas, extra_args=None, callback=None, disable=None)-> SampleData:
    sampler_name = "Unknown"

    for samplers_setting in samplers_k_diffusion:
        if samplers_setting[1] == sampler:
            sampler_name = samplers_setting[0]
    print(f"SDU sample_start sampler:{sampler}({sampler_name})\n",flush=True)
    if(sigmas is not None):
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
    extra_args = {} if extra_args is None else extra_args

    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    
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
    extra_args = {} if extra_args is None else extra_args

    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
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
    extra_args = {} if extra_args is None else extra_args
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    
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
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    
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
        if sigmas[i + 1] == 0:
            # Euler method
            dt = sigmas[i + 1] - sigma_hat
            sample_data.x = sample_data.x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigma_hat.log().lerp(sigmas[i + 1].log(), 0.5).exp()
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigmas[i + 1] - sigma_hat
            x_2 = sample_data.x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            sample_data.x = sample_data.x + d_2 * dt_2
    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_dpm_2_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    
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
        if sigma_down == 0:
            # Euler method
            dt = sigma_down - sigmas[i]
            sample_data.x = sample_data.x + d * dt
        else:
            # DPM-Solver-2
            sigma_mid = sigmas[i].log().lerp(sigma_down.log(), 0.5).exp()
            dt_1 = sigma_mid - sigmas[i]
            dt_2 = sigma_down - sigmas[i]
            x_2 = sample_data.x + d * dt_1
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d_2 = to_d(x_2, sigma_mid, denoised_2)
            sample_data.x = sample_data.x + d_2 * dt_2
            sample_data.x = sample_data.x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    sample_end(sample_data)#SDU Hijack
    return sample_data.x

class DPMSolver(nn.Module):
    """DPM-Solver. See https://arxiv.org/abs/2206.00927."""

    def __init__(self, model, sample_data:SampleData, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback
        self.sample_data = sample_data
    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        x_2 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, 'eps', x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1)
        u2 = x - self.sigma(s2) * (r2 * h).expm1() * eps - self.sigma(s2) * (r2 / r1) * ((r2 * h).expm1() / (r2 * h) - 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2)
        x_3 = x - self.sigma(t_next) * h.expm1() * eps - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        return x_3, eps_cache

    def dpm_solver_fast(self, sample_data_x, t_start, t_end, nfe, eta=0., s_noise=1., noise_sampler=None):
        self.sample_data.x = sample_data_x;
        noise_sampler = default_noise_sampler(self.sample_data.x) if noise_sampler is None else noise_sampler
        if not t_end > t_start and eta:
            raise ValueError('eta must be 0 for reverse sampling')

        m = math.floor(nfe / 3) + 1
        ts = torch.linspace(t_start, t_end, m + 1, device=self.sample_data.x.device)

        if nfe % 3 == 0:
            orders = [3] * (m - 2) + [2, 1]
        else:
            orders = [3] * (m - 1) + [nfe % 3]

        for i in range(len(orders)):
            #SDU Hijack Start
            self.sample_data.step = i
            global_setting.trigger(self.sample_data)
            if(global_setting.skip_sample(self.sample_data)):
                continue
            #SDU Hijack End
            eps_cache = {}
            t, t_next = ts[i], ts[i + 1]
            if eta:
                sd, su = get_ancestral_step(self.sigma(t), self.sigma(t_next), eta)
                t_next_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t_next) ** 2 - self.sigma(t_next_) ** 2) ** 0.5
            else:
                t_next_, su = t_next, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', self.sample_data.x, t)
            denoised = self.sample_data.x - self.sigma(t) * eps
            if self.info_callback is not None:
                self.info_callback({'x': self.sample_data.x, 'i': i, 't': ts[i], 't_up': t, 'denoised': denoised})

            if orders[i] == 1:
                self.sample_data.x, eps_cache = self.dpm_solver_1_step(self.sample_data.x, t, t_next_, eps_cache=eps_cache)
            elif orders[i] == 2:
                self.sample_data.x, eps_cache = self.dpm_solver_2_step(self.sample_data.x, t, t_next_, eps_cache=eps_cache)
            else:
                self.sample_data.x, eps_cache = self.dpm_solver_3_step(self.sample_data.x, t, t_next_, eps_cache=eps_cache)

            self.sample_data.x = self.sample_data.x + su * s_noise * noise_sampler(self.sigma(t), self.sigma(t_next))

        return self.sample_data.x

    def dpm_solver_adaptive(self, x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None):
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        if order not in {2, 3}:
            raise ValueError('order should be 2 or 3')
        forward = t_end > t_start
        if not forward and eta:
            raise ValueError('eta must be 0 for reverse sampling')
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety)
        info = {'steps': 0, 'nfe': 0, 'n_accept': 0, 'n_reject': 0}

        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            eps_cache = {}
            t = torch.minimum(t_end, s + pid.h) if forward else torch.maximum(t_end, s + pid.h)
            if eta:
                sd, su = get_ancestral_step(self.sigma(s), self.sigma(t), eta)
                t_ = torch.minimum(t_end, self.t(sd))
                su = (self.sigma(t) ** 2 - self.sigma(t_) ** 2) ** 0.5
            else:
                t_, su = t, 0.

            eps, eps_cache = self.eps(eps_cache, 'eps', x, s)
            denoised = x - self.sigma(s) * eps

            if order == 2:
                x_low, eps_cache = self.dpm_solver_1_step(x, s, t_, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_2_step(x, s, t_, eps_cache=eps_cache)
            else:
                x_low, eps_cache = self.dpm_solver_2_step(x, s, t_, r1=1 / 3, eps_cache=eps_cache)
                x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info['n_accept'] += 1
            else:
                info['n_reject'] += 1
            info['nfe'] += order
            info['steps'] += 1

            if self.info_callback is not None:
                self.info_callback({'x': x, 'i': info['steps'] - 1, 't': s, 't_up': s, 'denoised': denoised, 'error': error, 'h': pid.h, **info})

        return x, info

@torch.no_grad()
def sample_dpm_fast(model, x, sigma_min, sigma_max, n, extra_args=None, callback=None, disable=None, eta=0., s_noise=1., noise_sampler=None):
    """DPM-Solver-Fast (fixed step size). See https://arxiv.org/abs/2206.00927."""
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, None, extra_args, callback, disable)#SDU Hijack

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(total=n, disable=disable) as pbar:
        dpm_solver = DPMSolver(model, sample_data, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})

        sample_data.x = dpm_solver.dpm_solver_fast(sample_data.x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), n, eta, s_noise, noise_sampler)
        sample_end(sample_data)#SDU Hijack
        return sample_data.x


@torch.no_grad()
def sample_dpm_adaptive(model, x, sigma_min, sigma_max, extra_args=None, callback=None, disable=None, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0., icoeff=1., dcoeff=0., accept_safety=0.81, eta=0., s_noise=1., noise_sampler=None, return_info=False):
    """DPM-Solver-12 and 23 (adaptive step size). See https://arxiv.org/abs/2206.00927."""
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, None, extra_args, callback, disable)#SDU Hijack

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError('sigma_min and sigma_max must not be 0')
    with tqdm(disable=disable) as pbar:
        dpm_solver = DPMSolver(model, sample_data, extra_args, eps_callback=pbar.update)
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback({'sigma': dpm_solver.sigma(info['t']), 'sigma_hat': dpm_solver.sigma(info['t_up']), **info})
        x, info = dpm_solver.dpm_solver_adaptive(x, dpm_solver.t(torch.tensor(sigma_max)), dpm_solver.t(torch.tensor(sigma_min)), order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise, noise_sampler)

    sample_end(sample_data)#SDU Hijack
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2s_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
    extra_args = {} if extra_args is None else extra_args
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    
    noise_sampler = default_noise_sampler(sample_data.x) if noise_sampler is None else noise_sampler
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

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
        if sigma_down == 0:
            # Euler method
            d = to_d(sample_data.x, sigmas[i], denoised)
            dt = sigma_down - sigmas[i]
            sample_data.x = sample_data.x + d * dt
        else:
            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * sample_data.x - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            sample_data.x = (sigma_fn(t_next) / sigma_fn(t)) * sample_data.x - (-h).expm1() * denoised_2
        # Noise addition
        if sigmas[i + 1] > 0:
            sample_data.x = sample_data.x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_dpmpp_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=1 / 2):
    """DPM-Solver++ (stochastic)."""
    extra_args = {} if extra_args is None else extra_args
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(sample_data.x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

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
        if sigmas[i + 1] == 0:
            # Euler method
            d = to_d(sample_data.x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            sample_data.x = sample_data.x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)

            # Step 1
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * sample_data.x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            sample_data.x = (sigma_fn(t_next_) / sigma_fn(t)) * sample_data.x - (t - t_next_).expm1() * denoised_d
            sample_data.x = sample_data.x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su

    sample_end(sample_data)#SDU Hijack
    return sample_data.x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, solver_type='midpoint'):
    """DPM-Solver++(2M) SDE."""
    sample_data = sample_start(model, inspect.currentframe().f_code.co_name, x, sigmas, extra_args, callback, disable)#SDU Hijack
    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(sample_data.x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = sample_data.x.new_ones([sample_data.x.shape[0]])

    old_denoised = None
    h_last = None

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
        if sigmas[i + 1] == 0:
            # Denoising step
            sample_data.x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            sample_data.x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * sample_data.x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    sample_data.x = sample_data.x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    sample_data.x = sample_data.x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            sample_data.x = sample_data.x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h

    sample_end(sample_data)#SDU Hijack
    return sample_data.x