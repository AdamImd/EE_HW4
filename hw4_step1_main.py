import torch
import os
from step1_utils.models.unet import create_model
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import step1_utils.utils as utils
from step1_utils.DDIM_sampler import Sampler as DDIM
import argparse
import torchvision.transforms as transforms
from step1_utils.data.dataloader import get_dataset, get_dataloader
from step1_utils.degradations import GaussianNoise, get_degradation
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips


class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None
        
        # hyperparameters for path & dataset
        self.parser.add_argument('--out_path', type=str, default='results/step1_results', help='results file directory')
        self.parser.add_argument('--dataset', type=str, default='CelebA_HQ', help='either choose CelebA_HQ or ImageNet')
        self.parser.add_argument('--sigma_y', type=float, default=0.0, help='measurement noise')
        
        # hyperparameters for sampling
        self.parser.add_argument('--diff_timesteps', type=int, default=1000, help='Original number of steps from Ho et al. (2020) which is 1000 - do not change')
        self.parser.add_argument('--desired_timesteps', type=int, default=1000, help='How many steps do you want?')
        self.parser.add_argument('--eta', type=float, default=1.0, help='Should be between [0.0, 1.0]')
        self.parser.add_argument('--schedule', type=str, default="1000", help="regular/irregular schedule to use (jumps)")
        
        # hyperparameters for algos
        self.parser.add_argument('--ps_type', type=str, default="ILVR", help="choose from unconditional, ILVR, MCG, DDNM, DPS")
        self.parser.add_argument('--degradation', type=str, default='Inpainting', help='SR or Inpainting')
        self.parser.add_argument('--zeta_ilvr', type=float, default=1.0, help='ILVR weighting parameter (tune heuristically)')
        
        # hyperparameters for the inpainting mask & SR
        self.parser.add_argument('--mask_type', type=str, default="box", help='box or random')
        self.parser.add_argument('--random_amount', type=float, default=0.8, help='how much do you want to mask out?')
        self.parser.add_argument('--box_indices', type=int, nargs=4, default=[30,30,128,128], help='inpainting box indices - (y,x,height,width)')
        self.parser.add_argument('--scale_factor', type=int, default=4, help='SR scale factor')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        return self.conf

class posterior_samplers():
    def __init__(self, conf, sampler_operator, score_model):
        self.conf = conf
        self.sampler_operator = sampler_operator
        self.score_model = score_model
    
    def predict_x0_hat(self, x_t, t, model_output):
        alpha_t = utils.extract_and_expand(self.sampler_operator.alphas_cumprod, t, x_t)
        # x0_hat = (1/sqrt(alpha_bar_t)) * x_t - (sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t)) * epsilon
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = torch.sqrt(1.0 - alpha_t) / torch.sqrt(alpha_t)
        x0_hat = coeff1 * x_t - coeff2 * model_output 
        return utils.clip_denoised(x0_hat)
    
    def sample_ddim(self, x_t, t, x0_hat, model_output):
        sigma = (self.conf.eta * torch.sqrt((1 - self.sampler_operator.alphas_cumprod_prev) / (1 - self.sampler_operator.alphas_cumprod)) 
                 * torch.sqrt(1 - self.sampler_operator.alphas_cumprod / self.sampler_operator.alphas_cumprod_prev))
        posterior_mean_coef1 = torch.sqrt(self.sampler_operator.alphas_cumprod_prev)
        posterior_mean_coef2 = torch.sqrt(1.0 - self.sampler_operator.alphas_cumprod_prev - sigma**2)
        coef1 = utils.extract_and_expand(posterior_mean_coef1, t, x0_hat)
        coef2 = utils.extract_and_expand(posterior_mean_coef2, t, model_output)
        x_t_prev = coef1 * x0_hat + coef2 * model_output

        noise = torch.randn_like(x_t)
        # t is a tensor, so we need to check if any element is non-zero
        if t.any():
            x_t_prev = x_t_prev + utils.extract_and_expand(sigma, t, noise) * noise
        return x_t_prev
    
    def q_sample(self, data, t):
        ############################
        # TODO: Implement q(xt−1 | x0) = N (xt−1; √¯αt−1 x0, (1 − ¯αt−1)I)
        # Hint-1: Reparametrization Trick
        # Hint-2: You can get \bar{α}_{t−1} from --> self.sampler_operator.alphas_cumprod_prev
        ############################
        # Get alpha_bar_{t-1}
        alpha_bar_prev = utils.extract_and_expand(self.sampler_operator.alphas_cumprod_prev, t, data)
        # Reparametrization: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * noise
        noise = torch.randn_like(data)
        q_xt_x0 = torch.sqrt(alpha_bar_prev) * data + torch.sqrt(1 - alpha_bar_prev) * noise
        return q_xt_x0
    
    def ilvr(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement ILVR based on the HW PDF description
        
        # Hint-1: You can get the model output similar to HW3:
        # model_output = self.score_model(x_t, model_t)
        # model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        
        # Hint-2: A, A^T or A^\dagger operations can be performed by:
        # A_funcs.A(), A_funcs.At(), A_funcs.A_pinv()
        ############################
        # Step 1: Get model output (predicted noise)
        with torch.no_grad():
            model_output = self.score_model(x_t, model_t)
            model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
        
            # Step 2: Predict x0_hat from x_t (Tweedie denoised estimate)
            x0_hat = self.predict_x0_hat(x_t, t, model_output)
            
            # Step 3: DDIM sampling to get x'_{t-1}
            x_t_prev_prime = self.sample_ddim(x_t, t, x0_hat, model_output)
            
            # Step 4: Get y_{t-1} by noising the measurement through q_sample
            # y0 = A^{\dagger}(y) gives us the pseudo-inverse reconstruction
            y0 = A_funcs.A_pinv(measurement).reshape(x_t.shape)
            y_t_prev = self.q_sample(y0, t)
            
            # Step 5: ILVR update: x_{t-1} = x'_{t-1} + zeta * A^{\dagger}(y_{t-1} - A(x'_{t-1}))
            zeta_ilvr = self.conf.zeta_ilvr  # Tunable heuristically
            x_t_prev = x_t_prev_prime + zeta_ilvr * A_funcs.A_pinv(A_funcs.A(y_t_prev) - A_funcs.A(x_t_prev_prime)).reshape(x_t.shape)
        
        return x_t_prev
    
    def mcg(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement MCG based on the HW PDF description
        ############################
        x_t_prev = None
        return x_t_prev
    
    def ddnm(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DDNM based on the HW PDF description
        ############################
        x_t_prev = None
        return x_t_prev
    
    def dps(self, x_t, t, model_t, measurement, A_funcs):
        ############################
        # TODO: Implement DPS based on the HW PDF description
        ############################
        x_t_prev = None
        return x_t_prev
        
def main():
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    conf = Config().parse()
    
    print('*' * 60 + f'\nSTARTED DDIM Sampling with eta = \"%.1f\" \n' %conf.eta)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create and config model
    model_config = utils.load_yaml("step1_utils/models/" + conf.dataset + "_model_config.yaml")
    score_model = create_model(**model_config).to(device).eval()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(conf.dataset, f"step1_utils/data/{conf.dataset}/", transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    noiser = GaussianNoise(conf.sigma_y*2.0)
    A_funcs = get_degradation(conf, device)

    # Unconditional sampling for part (a) - generates 5 samples and saves as grid
    if conf.ps_type == "unconditional":
        num_samples = 5
        samples = []
        for i in range(num_samples):
            print(f'\nGenerating unconditional sample {i+1}/{num_samples}...')
            sampler_operator = DDIM(conf)
            x_t = utils.get_noise_x_t(device)
            pbar = list(range(conf.desired_timesteps))[::-1]
            time_map = sampler_operator.recreate_alphas().to(device)
            ps_ops = posterior_samplers(conf, sampler_operator, score_model)
            
            for idx in tqdm(pbar):
                time = torch.tensor([idx] * x_t.shape[0], device=device)
                # Get model output (predicted noise)
                with torch.no_grad():
                    model_output = score_model(x_t, time_map[time])
                    model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
                # Predict x0_hat and sample x_{t-1}
                x0_hat = ps_ops.predict_x0_hat(x_t, time, model_output)
                x_t = ps_ops.sample_ddim(x_t, time, x0_hat, model_output)
            
            samples.append(utils.clear_color(x_t))
            # Save individual sample
            sample_path = os.path.join(conf.out_path, conf.dataset, 'unconditional', f'sample_{i+1}.png')
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            plt.imsave(sample_path, samples[-1])
        return
    
    # Sampling (for conditional methods: ILVR, MCG, DDNM, DPS)
    for i, ref_img in enumerate(loader):
        print(f'\nSampling for Image {i+1} has started!')
        sampler_operator = DDIM(conf)
        ref_img = ref_img.to(device)
        measurement = noiser(A_funcs.A(ref_img))
        x_t = utils.get_noise_x_t(device)
        if conf.ps_type == "DPS":  # Only DPS needs gradients
            x_t = x_t.requires_grad_()
        pbar = (list(range(conf.desired_timesteps))[::-1])
        time_map = sampler_operator.recreate_alphas().to(device)
        ps_ops = posterior_samplers(conf, sampler_operator, score_model)
        
        for idx in tqdm(pbar):
            time = torch.tensor([idx] * x_t.shape[0], device=device)
            if conf.ps_type == "ILVR":
                x_t_prev = ps_ops.ilvr(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "MCG":
                x_t_prev = ps_ops.mcg(x_t, time, time_map[time], measurement, A_funcs)    
            elif conf.ps_type == "DDNM":
                x_t_prev = ps_ops.ddnm(x_t, time, time_map[time], measurement, A_funcs)         
            elif conf.ps_type == "DPS":
                x_t_prev = ps_ops.dps(x_t, time, time_map[time], measurement, A_funcs)
            else:
                raise ValueError(f"Unknown ps_type: {conf.ps_type}")
            x_t = x_t_prev
        
        # Compute metrics
        ref_np = utils.clear_color(ref_img)
        recon_np = utils.clear_color(x_t)
        meas_np = utils.clear_color(A_funcs.A_pinv(measurement).reshape(1,3,256,256))
        
        # PSNR and SSIM
        psnr_val = psnr(ref_np, recon_np, data_range=1.0)
        ssim_val = ssim(ref_np, recon_np, data_range=1.0, channel_axis=2)
        
        # LPIPS
        with torch.no_grad():
            lpips_model = lpips.LPIPS(net='alex').to(device)
            ref_tensor = torch.tensor(ref_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1  # Scale to [-1, 1]
            recon_tensor = torch.tensor(recon_np).permute(2,0,1).unsqueeze(0).to(device) * 2 - 1
            lpips_val = lpips_model(ref_tensor, recon_tensor).item()
        
        print(f'Image {i+1} - PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}')
        
        # Save image with metrics in filename
        deg_name = f"{conf.degradation}"
        if conf.degradation == "SR":
            deg_name += f"_x{conf.scale_factor}"
        elif conf.degradation == "Inpainting":
            deg_name += f"_{conf.mask_type}"
            if conf.mask_type == "random":
                deg_name += f"_{int(conf.random_amount*100)}pct"
        
        image_filename = f"recon_{i+1}.png"
        image_path = os.path.join(conf.out_path, conf.dataset, conf.ps_type, deg_name, image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Create figure with reference, measurement, and reconstruction
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ref_np)
        axes[0].set_title('Reference')
        axes[0].axis('off')
        axes[1].imshow(meas_np)
        axes[1].set_title('Measurement (A†y)')
        axes[1].axis('off')
        axes[2].imshow(recon_np)
        axes[2].set_title(f'Reconstruction\nPSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}')
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print('\nFINISHED Sampling!\n' + '*' * 60)

if __name__ == '__main__':
    main()