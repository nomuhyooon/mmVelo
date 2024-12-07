import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.nn.parameter import Parameter
from torch.nn import init
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.modules import VAE_RNA, VAE_ATAC
from mmvelo_multi.funcs import kl_divergence, convert_mean_disp_to_counts_logits, ZINB_logits


class DREG_PRE(pl.LightningModule):
    def __init__(self, rna_dim, atac_dim, r_h1_dim, r_h2_dim, a_h1_dim, a_h2_dim,
                 z_dim, d_h_dim, l_prior_r, l_prior_a, lr, z_learnable=True, d_coeff=1e-2, 
                 warmup=50, llik_scaling=True, pretrain_first_end=False):
        super().__init__()
        self.vaes = nn.ModuleList([
            VAE_RNA(rna_dim, r_h1_dim, r_h2_dim, z_dim, d_h_dim, l_prior_r, d_coeff=d_coeff),
            VAE_ATAC(atac_dim, a_h1_dim, a_h2_dim, z_dim, l_prior_a)
        ])
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=z_learnable)
        ])
        self._pd_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=False),
            nn.Parameter(torch.ones(1, z_dim) * torch.log(torch.exp(torch.tensor([1]))-1), requires_grad=z_learnable)
        ])
        self.rna_dim, self.atac_dim = rna_dim, atac_dim
        self.l_prior_r = l_prior_r
        self.l_prior_a = l_prior_a
        self.log_s = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_u = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_a = Parameter(torch.Tensor(atac_dim).to(torch.float32), requires_grad=True)
        self.log_beta = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_gamma = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.logvar_s = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.logvar_u = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.logvar_a = Parameter(torch.Tensor(atac_dim).to(torch.float32), requires_grad=True)
        
        self.reset_parameters()
        self.lr = lr
        self.d_coeff = d_coeff
        self.llik_scaling = llik_scaling
        self.warmup = warmup
        self.pretrain_first_end = pretrain_first_end
        
        if self.pretrain_first_end:
            self.set_grad_for_pretrain_second()
            
    def reset_parameters(self):
        init.normal_(self.log_s)
        init.normal_(self.log_u)
        init.normal_(self.log_a)
        init.zeros_(self.log_gamma)
        init.zeros_(self.log_beta)
        init.normal_(self.logvar_s)
        init.normal_(self.logvar_u)
        init.normal_(self.logvar_a)

    def set_norm_mat(self, dm):
        self.register_buffer("norm_mat_s", torch.tensor(dm.norm_mat_r[0]))
        self.register_buffer("norm_mat_u", torch.tensor(dm.norm_mat_r[1]))
        self.register_buffer("norm_mat_a", torch.tensor(dm.norm_mat_a))

    def set_retain_gene_idx(self, dm):
        if dm.retain_genes_idx is None:
            self.register_buffer("retain_gene_idx", torch.ones(self.rna_dim))
        else:
            retain_gene_idx = np.logical_not(dm.retain_genes_idx).astype(np.float32)
            self.register_buffer("retain_gene_idx", torch.tensor(retain_gene_idx))

    def set_grad_for_pretrain_second(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.vaes[0].dec_s.parameters():
            param.requires_grad = True
        for param in self.vaes[0].dec_u.parameters():
            param.requires_grad = True
        for param in self.vaes[1].dec_a.parameters():
            param.requires_grad = True
        self.logvar_s.requires_grad = True
        self.logvar_u.requires_grad = True
        self.logvar_a.requires_grad = True
        

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def pd_params(self):
        return self._pd_params[0], F.softplus(self._pd_params[1])
    
    def forward(self, batch):
        if self.pretrain_first_end:
            s, u, a, _, _, _ = batch
        else:
            s, u, a = batch
        qz_mu_r, qz_logvar_r = self.vaes[0].enc_z(s, u)
        qz_r = dist.Normal(qz_mu_r, F.softplus(qz_logvar_r))
        qz_mu_a, qz_logvar_a = self.vaes[1].enc_z(a)
        qz_a = dist.Normal(qz_mu_a, F.softplus(qz_logvar_a))
        z_moe = (qz_r.mean + qz_a.mean) / 2

        ql_mu, ql_logvar = self.vaes[0].enc_l(s+u)
        ql_r = dist.log_normal.LogNormal(ql_mu, F.softplus(ql_logvar))
        l_r = ql_r.mean / self.l_prior_r.mean
        ql_mu, ql_logvar = self.vaes[1].enc_l(a)
        ql_a = dist.log_normal.LogNormal(ql_mu, F.softplus(ql_logvar))
        l_a = ql_a.mean / self.l_prior_a.mean

        s_raw, u_raw = self.vaes[0].dec_su(z_moe)
        a_params = self.vaes[1].dec_ald(z_moe)
        
        s_ld = s_raw[0] * l_r * self.norm_mat_s
        u_ld = u_raw[0] * l_r * self.norm_mat_u * self.retain_gene_idx
        a_ld = a_params[0] * l_a * self.norm_mat_a

        return z_moe, s_ld, u_ld, a_ld, ql_r.mean, ql_a.mean, s_raw[0] * self.norm_mat_s, u_raw[0] * self.norm_mat_u, a_params[0] * self.norm_mat_a

    def compute_kl_divergence(self, qz, ql):
        kld_z = kl_divergence(qz, dist.Normal(*self.pz_params)).sum(-1)
        kld_l = kl_divergence(ql, self.l_prior).sum(-1)
        return kld_z, kld_l

    def compute_zinb_loss_s(self, x, ld, l, zinb=True):
        ld, gate_logits = ld[0], ld[1]
        counts, logits = convert_mean_disp_to_counts_logits(ld * l * self.norm_mat_s, torch.exp(self.log_s))
        if zinb:
            px_z = ZINB_logits(counts, logits=logits, gate_logits=gate_logits)
        else:
            px_z = dist.NegativeBinomial(counts, logits=logits)
        lpx_z = px_z.log_prob(x)
        return lpx_z.sum(-1)

    def compute_zinb_loss_u(self, x, ld, l, zinb=True):
        ld, gate_logits = ld[0], ld[1]
        counts, logits = convert_mean_disp_to_counts_logits(ld * l * self.norm_mat_u, torch.exp(self.log_u))
        if zinb:
            px_z = ZINB_logits(counts, logits=logits, gate_logits=gate_logits)
        else:
            px_z = dist.NegativeBinomial(counts, logits=logits)
        lpx_z = px_z.log_prob(x) * self.retain_gene_idx
        #lpx_z[:, self.retain_gene_idx==0] = 0
        return lpx_z.sum(-1)

    def compute_zinb_loss_a(self, x, a_params, l, zinb=True):
        a_raw, a_gate = a_params[0], a_params[1]
        counts, logits = convert_mean_disp_to_counts_logits(a_raw * l * self.norm_mat_a, torch.exp(self.log_a))
        if zinb:
            px_z = ZINB_logits(counts, logits=logits, gate_logits=a_gate)
        else:
            px_z = dist.NegativeBinomial(counts, logits=logits)
        #px_z = dist.negative_binomial.NegativeBinomial(counts, logits=logits)
        lpx_z = px_z.log_prob(x)
        return lpx_z.sum(-1)

    def compute_normal_loss(self, x, ld, s_=False, u_=False, a_=False):
        ld, _ = ld[0], ld[1]
        if s_:
            px_z = dist.normal.Normal(ld * self.norm_mat_s, torch.exp(self.logvar_s).clamp(min=1e-2))

        elif u_:
            px_z = dist.normal.Normal(ld * self.norm_mat_u, torch.exp(self.logvar_u).clamp(min=1e-2))
            lpx_z = px_z.log_prob(x) * self.retain_gene_idx
            return lpx_z.sum(-1)
        
        elif a_:
            px_z = dist.normal.Normal(ld * self.norm_mat_a, torch.exp(self.logvar_a).clamp(min=1e-2))
        else:
            raise AssertionError()
        lpx_z = px_z.log_prob(x)
        return lpx_z.sum(-1)


    def compute_elbo_loss_warmup(self, batch):
        s, u, a = batch
        zs, ls, qz_xs, ql_xs = [], [], [], []
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            if m == 0:
                z_i, l_i, qz_x, ql, s_raw, u_raw = vae.forward(s, u)
                zs.append(z_i)
                ls.append(l_i)
                qz_xs.append(qz_x)
                ql_xs.append(ql)
                px_zs[m][m] = [s_raw, u_raw]
            elif m == 1:
                z_i, l_i, qz_x, ql, a_params = vae.forward(a)
                zs.append(z_i)
                ls.append(l_i)
                qz_xs.append(qz_x)
                ql_xs.append(ql)
                px_zs[m][m] = [a_params]

        for e, z in enumerate(zs): # z: sample x batch x dim
            for f, vae in enumerate(self.vaes):
                if e != f:
                    if f == 0:
                        s_raw, u_raw = vae.dec_su(z)
                        px_zs[e][f] = [s_raw, u_raw]
                    else:
                        a_params = vae.dec_ald(z)
                        px_zs[e][f] = [a_params]

        l_rna = ls[0] / self.l_prior_r.mean
        l_atac = ls[1] / self.l_prior_a.mean
        lpx_zs, klds_z, klds_l = [], [], []

        for m in range(len(self.vaes)):
            lnps_z = self.compute_zinb_loss_s(s, px_zs[m][0][0], l_rna, zinb=False)
            lnpu_z = self.compute_zinb_loss_u(u, px_zs[m][0][1], l_rna, zinb=False)
            lnpa_z = self.compute_zinb_loss_a(a, px_zs[m][1][0], l_atac, zinb=True)
            if self.llik_scaling:
                lnpa_z = lnpa_z * (self.rna_dim * 2 / self.atac_dim)
            kld_z = kl_divergence(qz_xs[m], dist.Normal(*self.pz_params))
            if m==0:
                kld_l = kl_divergence(ql_xs[m], self.l_prior_r)
            elif m==1:
                kld_l = kl_divergence(ql_xs[m],self.l_prior_a)
            lpx_zs.append(lnps_z + lnpu_z + lnpa_z)
            klds_z.append(kld_z.sum(-1))
            klds_l.append(kld_l.sum(-1))
        elbo_warmup = (1 / len(self.vaes)) *  (torch.stack(lpx_zs).sum(0) - ((self.current_epoch + 1) / self.warmup) * (torch.stack(klds_z).sum(0) + len(self.vaes) * torch.stack(klds_l).sum(0)))
        return -1 * elbo_warmup.mean(0)

    def compute_elbo_loss_dec(self, batch):
        s, u, a, ms, mu, ma = batch
        zs, _, qz_xs, _ = [], [], [], []
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            if m == 0:
                z_i, _, qz_x, _, s_raw, u_raw = vae.forward(s, u)
                zs.append(z_i)
                qz_xs.append(qz_x)
                px_zs[m][m] = [s_raw, u_raw]
            elif m == 1:
                z_i, _, qz_x, _, a_params = vae.forward(a)
                zs.append(z_i)
                qz_xs.append(qz_x)
                px_zs[m][m] = [a_params]

        for e, z in enumerate(zs): # z: sample x batch x dim
            for f, vae in enumerate(self.vaes):
                if e != f:
                    if f == 0:
                        s_raw, u_raw = vae.dec_su(z)
                        px_zs[e][f] = [s_raw, u_raw]
                    else:
                        a_params = vae.dec_ald(z)
                        px_zs[e][f] = [a_params]

        lpx_zs, klds_z = [], []

        for m in range(len(self.vaes)):
            lnps_z = self.compute_normal_loss(ms, px_zs[m][0][0], s_=True)
            lnpu_z = self.compute_normal_loss(mu, px_zs[m][0][1], u_=True)
            lnpa_z = self.compute_normal_loss(ma, px_zs[m][1][0], a_=True)
            if self.llik_scaling:
                lnpa_z = lnpa_z * (self.rna_dim * 2 / self.atac_dim)
            lpx_zs.append(lnps_z + lnpu_z + lnpa_z)
            kld_z = kl_divergence(qz_xs[m], dist.Normal(*self.pz_params))
            klds_z.append(kld_z.sum(-1))

        elbo = (1 / len(self.vaes)) *  torch.stack(lpx_zs).sum(0) - torch.stack(klds_z).sum(0)
        return -1 * elbo.mean(0)
            
    def training_step(self, batch, batch_idx):
        if self.pretrain_first_end:
            elbo_loss = self.compute_elbo_loss_dec(batch)
        else:
            elbo_loss = self.compute_elbo_loss_warmup(batch)
        self.log_dict({
                "elbo_loss": elbo_loss, 
            })
        return elbo_loss

    def validation_step(self, batch, batch_idx):
        if self.pretrain_first_end:
            elbo_loss = self.compute_elbo_loss_dec(batch)
        else:
            elbo_loss = self.compute_elbo_loss_warmup(batch)
        self.log_dict({
                "val_elbo_loss": elbo_loss, 
            })
        return elbo_loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_elbo_loss"}


class DREG_DYN(pl.LightningModule):
    def __init__(self, rna_dim, atac_dim, r_h1_dim, r_h2_dim, a_h1_dim, a_h2_dim,
                 z_dim, d_h_dim, l_prior_r, l_prior_a, lr, z_learnable=True, d_coeff=1e-2, 
                 warmup=50, filter_idx=None):
        super().__init__()
        self.rna_dim, self.atac_dim, self.z_dim = rna_dim, atac_dim, z_dim
        self.vaes = nn.ModuleList([
            VAE_RNA(rna_dim, r_h1_dim, r_h2_dim, z_dim, d_h_dim, l_prior_r, d_coeff=d_coeff),
            VAE_ATAC(atac_dim, a_h1_dim, a_h2_dim, z_dim, l_prior_a)
        ])
        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=False),
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=z_learnable)
        ])
        self._pd_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, z_dim), requires_grad=False),
            nn.Parameter(torch.ones(1, z_dim) * torch.log(torch.exp(torch.tensor([1]))-1), requires_grad=z_learnable)
        ])
        self.log_s = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_u = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_a = Parameter(torch.Tensor(atac_dim).to(torch.float32), requires_grad=True)
        self.logvar_s = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.logvar_u = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.logvar_a = Parameter(torch.Tensor(atac_dim).to(torch.float32), requires_grad=True)
        self.log_beta = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.log_gamma = Parameter(torch.Tensor(rna_dim).to(torch.float32), requires_grad=True)
        self.lr = lr
        self.d_coeff = d_coeff
        self.warmup = warmup
        self.reset_parameters()
        self.compute_dadt = False
        
    def reset_parameters(self):
        init.normal_(self.log_s)
        init.normal_(self.log_u)
        init.normal_(self.log_a)
        init.zeros_(self.log_gamma)
        init.zeros_(self.log_beta)
        init.normal_(self.logvar_s)
        init.normal_(self.logvar_u)

    def set_norm_mat(self, dm):
        self.register_buffer("norm_mat_s", torch.tensor(dm.norm_mat_r[0]))
        self.register_buffer("norm_mat_u", torch.tensor(dm.norm_mat_r[1]))
        self.register_buffer("norm_mat_a", torch.tensor(dm.norm_mat_a))

    def set_beta_gamma_ss_ratio(self):
        log_gamma_beta_ss = torch.tensor(self.log_gamma) - torch.tensor(self.log_beta) 
        self.register_buffer("log_gamma_beta_ss", torch.tensor(log_gamma_beta_ss))
    
    def set_filter_idx(self, filter_idx):
        self.register_buffer("filter_idx", filter_idx)

    def set_retain_gene_idx(self, dm):
        if dm.retain_genes_idx is None:
            self.register_buffer("retain_gene_idx", torch.ones(self.rna_dim))
        else:
            retain_gene_idx = np.logical_not(dm.retain_genes_idx).astype(np.float32)
            self.register_buffer("retain_gene_idx", torch.tensor(retain_gene_idx))


    def set_grad_for_training(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.vaes[0].enc_d.parameters():
            param.requires_grad = True
        self.log_gamma.requires_grad = True
        self.log_beta.requires_grad = True
        self._pd_params[1].requires_grad = True
    
    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1)

    @property
    def pd_params(self):
        return self._pd_params[0], F.softplus(self._pd_params[1])

    def get_params(self):
        z_var_ = [var.item() for var in self.pz_params[1][0]]
        d_var_ = [var.item() for var in self.pd_params[1][0]]

        kinetic_param_dev_ = self.compute_kinetic_param_reg().sum().item()
        num_estimated_genes_ = self.filter_idx.sum().item()

        return dict(z_var = z_var_, d_var = d_var_, 
                    kinetic_param_dev = kinetic_param_dev_,
                    num_estimated_genes = num_estimated_genes_)
    
    def forward(self, batch):
        s, u, a, _, _, _ = batch
        qz_mu_r, qz_logvar_r = self.vaes[0].enc_z(s, u)
        qz_r = dist.Normal(qz_mu_r, F.softplus(qz_logvar_r))
        qz_mu_a, qz_logvar_a = self.vaes[1].enc_z(a)
        qz_a = dist.Normal(qz_mu_a, F.softplus(qz_logvar_a))
        z_moe = (qz_r.mean + qz_a.mean) / 2
        qd = self.vaes[0].enc_dyn(z_moe)
        d_moe = qd.mean
        d_var = qd.scale

        s_raw, u_raw = self.vaes[0].dec_su(z_moe)
        s_raw, u_raw = s_raw[0], u_raw[0]
        s_raw, u_raw = s_raw * self.norm_mat_s, u_raw * self.norm_mat_u
        s_raw_dt, u_raw_dt = self.vaes[0].dec_su(z_moe + self.d_coeff * d_moe)
        s_raw_dt, u_raw_dt = s_raw_dt[0], u_raw_dt[0]
        s_raw_dt, u_raw_dt = s_raw_dt * self.norm_mat_s, u_raw_dt * self.norm_mat_u
        dsdt, dudt = s_raw_dt - s_raw, u_raw_dt - u_raw
        dudt = dudt * self.retain_gene_idx
        #a_params = self.vaes[1].dec_ald(z_moe)
        #a_params_dt = self.vaes[1].dec_ald(z_moe + self.d_coeff * d_moe)
        #dsdt_obs = torch.exp(-self.log_gamma_beta + self.log_gamma) * u_raw - torch.exp(self.log_gamma) * s_raw
        dsdt_obs = torch.exp(self.log_beta) * u_raw - torch.exp(self.log_gamma) * s_raw
        dsdt_obs = dsdt_obs * self.retain_gene_idx * self.filter_idx

        
        if self.compute_dadt:
            a_raw = self.vaes[1].dec_ald(z_moe)[0] * self.norm_mat_a
            a_raw_dt = self.vaes[1].dec_ald(z_moe + self.d_coeff * d_moe)[0] * self.norm_mat_a
            dadt = (a_raw_dt - a_raw)
        else:
            dadt = torch.zeros(1, 1)
        d_moe = d_moe * self.d_coeff

        return z_moe, d_moe, dsdt, dsdt_obs, d_var, dadt, dudt

    def compute_kl_divergence(self, qz, ql):
        kld_z = kl_divergence(qz, dist.Normal(*self.pz_params)).sum(-1)
        kld_l = kl_divergence(ql, self.l_prior).sum(-1)
        return kld_z, kld_l
    
    def compute_squared_wasserstein_distance(self, qd):
        return (qd.mean**2 +  qd.variance)


    def compute_cossim_loss_dsdt(self, ld, kappa=1):
        s_t, u_t, s_t_dt = ld[0][0], ld[1][0], ld[2][0]
        mu = (s_t_dt - s_t) * self.norm_mat_s
        
        #dsdt = (torch.exp(-self.log_gamma_beta + self.log_gamma) * u_t * self.norm_mat_u) - torch.exp(self.log_gamma) * s_t * self.norm_mat_s
        dsdt = (torch.exp(self.log_beta) * u_t * self.norm_mat_u) - torch.exp(self.log_gamma) * s_t * self.norm_mat_s

        if self.filter_idx is not None:
            
            mu = mu * self.filter_idx
            dsdt = dsdt * self.filter_idx

        mu = F.tanh(mu * self.retain_gene_idx) 
        dsdt = F.tanh(dsdt * self.retain_gene_idx)

        #mu = mu * self.retain_gene_idx
        #dsdt = dsdt * self.retain_gene_idx

        cos = nn.CosineSimilarity(dim=1)
        cossim = cos(mu, dsdt)

        scaling = self.rna_dim / self.z_dim
        return cossim * kappa * scaling
    
    
    def compute_kinetic_param_reg(self):
        log_gamma_beta_ss = self.log_gamma_beta_ss
        log_gamma_beta = self.log_gamma - self.log_beta
        reg = (log_gamma_beta - log_gamma_beta_ss) **2        
        return reg

    def compute_elbo_loss_warmup(self, batch):
        s, u, a, ms, mu, ma = batch
        zs, ds, qz_xs, qds = [], [], [], []
        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            if m == 0:
                z_i, l_i, d_i, qz_x, ql, qd, s_raw, u_raw, s_raw_dt, u_raw_dt = vae.forward_d(s, u, z_sample=False)
                zs.append(z_i)
                ds.append(d_i)
                qz_xs.append(qz_x)
                qds.append(qd)
                px_zs[m][m] = [s_raw, u_raw, s_raw_dt, u_raw_dt]
            elif m == 1:
                z_i, l_i, qz_x, ql, a_params = vae.forward(a, z_sample=False)
                qd = self.vaes[0].enc_dyn(z_i)
                d_i = qd.rsample()
                zs.append(z_i)
                ds.append(d_i)
                qz_xs.append(qz_x)
                qds.append(qd)
                px_zs[m][m] = [a_params]

        for e, z in enumerate(zs): # z: sample x batch x dim
            for f, vae in enumerate(self.vaes):
                if e != f:
                    if f == 0:
                        s_raw, u_raw = vae.dec_su(z)
                        s_raw_dt, u_raw_dt = vae.dec_su(z + ds[e] * self.d_coeff)
                        px_zs[e][f] = [s_raw, u_raw, s_raw_dt, u_raw_dt]
                    else:
                        a_params = vae.dec_ald(z)
                        px_zs[e][f] = [a_params]

        lpx_zs, klds_z, klds_d = [], [], []

        for m in range(len(self.vaes)):
            lnp_dsdt_zd = self.compute_cossim_loss_dsdt(px_zs[m][0])
            kld_z = kl_divergence(qz_xs[m], dist.Normal(*self.pz_params))
            kld_d = kl_divergence(qds[m], dist.Normal(*self.pd_params))
            ## squared wasserstein distance
            ##kld_d = self.compute_squared_wasserstein_distance(qds[m])
            lpx_zs.append(lnp_dsdt_zd)
            klds_z.append(kld_z.sum(-1))
            klds_d.append(kld_d.sum(-1))
        elbo_warmup = (1 / len(self.vaes)) *  (torch.stack(lpx_zs).sum(0) - ((self.current_epoch + 1) / self.warmup) * (torch.stack(klds_z).sum(0) + torch.stack(klds_d).sum(0)))
        return -1 * elbo_warmup.mean(0) + self.compute_kinetic_param_reg().sum()
    

    def training_step(self, batch, batch_idx):
        elbo_loss = self.compute_elbo_loss_warmup(batch)
        self.log_dict({
            "elbo_loss_d": elbo_loss, 
        })
        return elbo_loss

    def validation_step(self, batch, batch_idx):
        elbo_loss = self.compute_elbo_loss_warmup(batch)
        self.log_dict({
            "val_elbo_loss_d": elbo_loss, 
        })
        return elbo_loss

    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    """
    
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_elbo_loss_d"}


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except not watching the first `warmup` epochs.
    """
    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < self.warmup:
            return
        else:
            super()._run_early_stopping_check(trainer)