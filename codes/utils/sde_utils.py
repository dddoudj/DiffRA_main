import math
import torch
import abc
import torch.optim as optim
from tqdm import tqdm
import torchvision.utils as tvutils
from torch.autograd import Variable
import open_clip
import torch.nn as nn
import os
from scipy import integrate
import torchvision
import timm

import pretrainedmodels
from torch.autograd import Variable as V
import torch.nn.functional as F

'''from torch_nets import (
    tf2torch_inception_v4,
    tf2torch_inception_v3,
    tf2torch_resnet_v2_101,
    tf2torch_inc_res_v2,
    tf2torch_ens_adv_inc_res_v2,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_adv_inception_v3,
)'''


class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class TfNormalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode



classifier = torchvision.models.inception_v3(pretrained=True).eval().cuda()


clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")


target_layer0 = 'Mixed_7c'
target_layer = [target_layer0]
mean, stddev = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
criterion_gcam = nn.MSELoss(reduction='mean')

# 图像变换：用于修改图像尺寸和增广数据，同时归一化数据，以使数据能够适配CLIP模型
tfms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪
        torchvision.transforms.RandomAffine(5),  # 随机扭曲图片
        torchvision.transforms.RandomHorizontalFlip(),  # 随机左右镜像
        torchvision.transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def ode_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t)

    def reverse_sde_step(self, x, score, t):
        return x - self.sde_reverse_drift(x, score, t) - self.dispersion(x, t)

    def reverse_ode_step(self, x, score, t):
        return x - self.ode_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)

        return x


#############################################################################


class IRSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, max_sigma, T=100, schedule='cosine', eps=0.01, device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma >= 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule, eps)

    def _initialize(self, max_sigma, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2  # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]  # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    #####################################

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, t):
        return self.thetas[t] * (self.mu - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt

    def ode_reverse_drift(self, x, score, t):
        return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t] ** 2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        noise = self.model(x, self.mu, t, **kwargs)
        return self.get_score_from_noise(noise, t)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - self.mu) + term2 * (x0 - self.mu) + self.mu

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t) ** 2

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * 0.8)  # 0.9

        if 0.8 < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        return padded if torch.rand(1) < 0.9 else x

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x



    # 定义一个损失函数，用于获取图片的特征，然后与提示文字的特征进行对比
    def clip_loss(self, image, text_features):
        image_features = clip_model.encode_image(tfms(image))  # 注意施加上面定义好的变换
        input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
        # 使用Squared Great Circle Distance计算距离
        dists = (input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2))
        return dists.mean()

    def reverse_sde(self, xt, x0, T=-1, save_states=False, save_dir='sde_state', **kwargs):  # guide
        clip_model.to(self.device)
        T = self.T if T < 0 else T
        x = xt.clone()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        with open('imagenet_classes2.txt', errors='ignore') as f:
            labels = [line.strip() for line in f.readlines()]
            x_pre = classifier(x0)
            # print(x_pre)
            pred_index = torch.argmax(x_pre, 1).cpu().detach().numpy()
            string = labels[pred_index[0]]
            index = string.find(",")
            prompt = string[:index]
            # prompt = f'"a photo of {prompt}"'
            prompt = f'"{prompt}"'
            print(prompt)

            '''a, idxadv = torch.sort(x_pre, descending=True)
            pred_indexadv = idxadv[0][1].cpu().detach().numpy()
            stringadv = labels[pred_indexadv]
            indexadv = stringadv.find(",")
            promptadv = stringadv[:indexadv]
            promptadv = f'"{promptadv}"'
            print(promptadv)
        textadv = open_clip.tokenize([promptadv]).to(self.device)'''
        text = open_clip.tokenize([prompt]).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_model.encode_text(text)
            # text_featuresadv = clip_model.encode_text(textadv)

        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                #for m in tqdm(reversed(range(1, t + 1))):
                xclone = self.reverse_sde_step(x, score, t)

                attackloss = loss_fn(classifier(xclone), x_pre)
                cliploss = self.clip_loss(xclone, text_features)
                # cliplossadv = self.clip_loss(xclone, text_featuresadv)

                loss = 0.002 * attackloss + 10 * cliploss
                cond_grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + cond_grad
            x = self.reverse_sde_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 15
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        pred0 = classifier(x)
        x_rpre0 = classifier(x0)
        pred_indexadv0 = torch.argmax(pred0, 1).cpu().detach().numpy()
        pred_indexr0 = torch.argmax(x_rpre0, 1).cpu().detach().numpy()

        if pred_indexadv0 != pred_indexr0:
            success0 = 1
        else:
            success0 = 0


        print("the GT current predict class name: %s" % pred_indexr0[0])
        prob = F.softmax(pred0, dim=1).max(1)
        print("the advsde current predict class name: %s" % pred_indexadv0[0])
        print(prob)

        return x, success0


    def reverse_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t, **kwargs)
            x = self.reverse_ode_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # sample ode using Black-box ODE solver (not used)
    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, ):
        shape = xt.shape

        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                       rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)

        self.set_mu(mu)

        batch = x0.shape[0]

        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        state_mean = self.mu_bar(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + state_mean

        return timesteps, noisy_states.to(torch.float32)

    def noise_state(self, tensor):
        return tensor + torch.randn_like(tensor) * self.max_sigma


################################################################################
################################################################################
############################ Denoising SDE ##################################
################################################################################
################################################################################


class DenoisingSDE(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, max_sigma, T, schedule='cosine', device=None):
        super().__init__(T, device)
        self.max_sigma = max_sigma / 255 if max_sigma > 1 else max_sigma
        self._initialize(self.max_sigma, T, schedule)

    def _initialize(self, max_sigma, T, schedule, eps=0.04):

        def linear_beta_schedule(timesteps):
            timesteps = timesteps + 1
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_beta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
            """
            timesteps = timesteps + 2
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            # betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(max_sigma ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(max_sigma ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_beta_schedule(T)
        else:
            thetas = linear_beta_schedule(T)
        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)
        self.sigma_bars = sigma_bars.to(self.device)

        self.mu = 0.
        self.model = None

    # set noise model for reverse process
    def set_model(self, model):
        self.model = model

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def mu_bar(self, x0, t):
        return x0

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def drift(self, x, x0, t):
        return self.thetas[t] * (x0 - x) * self.dt

    def sde_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t] ** 2 * (1 + A) * score * self.dt

    def ode_reverse_drift(self, x, score, t):
        A = torch.exp(-2 * self.thetas_cumsum[t] * self.dt)
        return -0.5 * self.sigmas[t] ** 2 * A * score * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        return -noise / self.sigma_bar(t)

    def get_init_state_from_noise(self, x, noise, t):
        return x - self.sigma_bar(t) * noise

    def get_init_state_from_score(self, x, score, t):
        return x + self.sigma_bar(t) ** 2 * score

    def score_fn(self, x, t):
        # need to preset the score_model
        noise = self.model(x, t)
        return self.get_score_from_noise(noise, t)

    ############### reverse sampling ################

    def get_real_noise(self, xt, x0, t):
        return (xt - self.mu_bar(x0, t)) / self.sigma_bar(t)

    def get_real_score(self, xt, x0, t):
        return -(xt - self.mu_bar(x0, t)) / self.sigma_bar(t) ** 2

    def reverse_sde(self, xt, x0=None, T=-1, save_states=False, save_dir='sde_state'):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            if x0 is not None:
                score = self.get_real_score(x, x0, t)
            else:
                score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    # 定义一个损失函数，用于获取图片的特征，然后与提示文字的特征进行对比
    def clip_loss(self, image, text_features):
        image_features = clip_model.encode_image(tfms(image))  # 注意施加上面定义好的变换
        input_normed = torch.nn.functional.normalize(image_features.unsqueeze(1), dim=2)
        embed_normed = torch.nn.functional.normalize(text_features.unsqueeze(0), dim=2)
        # 使用Squared Great Circle Distance计算距离
        dists = (input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2))
        return dists.mean()

    def reverse_ode(self, xt, x0, T=-1, save_states=False, save_dir='ode_state'):
        clip_model.to(self.device)
        T = self.T if T < 0 else T
        x = xt.clone()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        with open('imagenet_classes2.txt', errors='ignore') as f:
            labels = [line.strip() for line in f.readlines()]
            x_pre = classifier(x0)
            # print(x_pre)
            pred_index = torch.argmax(x_pre, 1).cpu().detach().numpy()
            string = labels[pred_index[0]]
            index = string.find(",")
            prompt = string[:index]
            # prompt = f'"a photo of {prompt}"'
            prompt = f'"{prompt}"'
            print(prompt)
        text = open_clip.tokenize([prompt]).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_model.encode_text(text)

        for t in tqdm(reversed(range(1, T + 1))):
            '''if x0 is not None:
                real_score = self.get_real_score(x, x0, t)'''

            score = self.score_fn(x, t)
            with torch.enable_grad():
                x = x.detach().requires_grad_()
                for m in tqdm(reversed(range(1, t + 1))):
                    xclone = self.reverse_sde_step(x, score, t)

                attackloss = loss_fn(classifier(xclone), x_pre)
                cliploss = self.clip_loss(xclone, text_features)
                loss = 0.00008 * attackloss + 1 * cliploss
                cond_grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + cond_grad
            x = self.reverse_ode_step(x, score, t)

            if save_states:
                interval = self.T // 100
                if t % interval == 0:
                    state = x.clone()
                    '''if x0 is not None:
                        state = torch.cat([x, score, real_score], dim=0)'''
                    os.makedirs(save_dir, exist_ok=True)
                    idx = t // interval
                    tvutils.save_image(state.data, f'{save_dir}/state_{idx}.png', normalize=False)

        pred0 = classifier(x)
        x_rpre0 = classifier(x0)
        pred_indexadv0 = torch.argmax(pred0, 1).cpu().detach().numpy()
        pred_indexr0 = torch.argmax(x_rpre0, 1).cpu().detach().numpy()

        if pred_indexadv0 != pred_indexr0:
            success0 = 1
        else:
            success0 = 0

        print("the GT current predict class name: %s" % pred_indexr0[0])
        prob = F.softmax(pred0, dim=1).max(1)
        print("the advsde current predict class name: %s" % pred_indexadv0[0])
        print(prob)

        return x, success0

    def ode_sampler(self, xt, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, ):
        shape = xt.shape

        def to_flattened_numpy(x):
            """Flatten a torch tensor `x` and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
            return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x):
            t = int(t)
            x = from_flattened_numpy(x, shape).to(self.device).type(torch.float32)
            score = self.score_fn(x, t)
            drift = self.ode_reverse_drift(x, score, t)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(xt),
                                       rtol=rtol, atol=atol, method=method)

        x = torch.tensor(solution.y[:, -1]).reshape(shape).to(self.device).type(torch.float32)

        return x

    def get_optimal_timestep(self, sigma, eps=1e-6):
        sigma = sigma / 255 if sigma > 1 else sigma
        thetas_cumsum_hat = -1 / (2 * self.dt) * math.log(1 - sigma ** 2 / self.max_sigma ** 2 + eps)
        T = torch.argmin((self.thetas_cumsum - thetas_cumsum_hat).abs())
        return T

    ##########################################################
    ########## below functions are used for training #########
    ##########################################################

    def reverse_optimum_step(self, xt, x0, t):
        A = torch.exp(-self.thetas[t] * self.dt)
        B = torch.exp(-self.thetas_cumsum[t] * self.dt)
        C = torch.exp(-self.thetas_cumsum[t - 1] * self.dt)

        term1 = A * (1 - C ** 2) / (1 - B ** 2)
        term2 = C * (1 - A ** 2) / (1 - B ** 2)

        return term1 * (xt - x0) + term2 * (x0 - x0) + x0

    def optimal_reverse(self, xt, x0, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            x = self.reverse_optimum_step(x, x0, t)

        return x

    def weights(self, t):
        # return 0.1 + torch.exp(-self.thetas_cumsum[t] * self.dt)
        return self.sigmas[t] ** 2

    def generate_random_states(self, x0):
        x0 = x0.to(self.device)

        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T + 1, (batch, 1, 1, 1)).long()

        noises = torch.randn_like(x0, dtype=torch.float32)
        noise_level = self.sigma_bar(timesteps)
        noisy_states = noises * noise_level + x0

        return timesteps, noisy_states

