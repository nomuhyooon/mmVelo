import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist


class LinearGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearGELU, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim, elementwise_affine=False),
            nn.GELU(),
            )

    def forward(self, x):
        h = self.f(x)
        return h

class Encoder_su(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim):
        super(Encoder_su, self).__init__()
        self.x_dim=x_dim
        self.s2h = LinearGELU(x_dim, h1_dim)
        self.u2h = LinearGELU(x_dim, h1_dim)
        self.h2h = LinearGELU(h1_dim*2, h2_dim)
        self.h2mu = nn.Linear(h2_dim, z_dim)
        self.h2logvar = nn.Linear(h2_dim, z_dim)

    def forward(self, s, u):
        h_s = self.s2h(s)
        h_u = self.u2h(u)
        h = torch.cat((h_s, h_u), dim=-1)
        h = self.h2h(h)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        return mu, logvar

class Encoder_a(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim):
        super(Encoder_a, self).__init__()
        self.x_dim=x_dim
        self.x2h = LinearGELU(x_dim, h1_dim)
        self.h2h = LinearGELU(h1_dim, h2_dim)
        self.h2mu = nn.Linear(h2_dim, z_dim)
        self.h2logvar = nn.Linear(h2_dim, z_dim)

    def forward(self, a):
        h = self.x2h(a)
        h = self.h2h(h)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        return mu, logvar

class Decoder_r(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder_r, self).__init__()
        self.z2h = LinearGELU(z_dim, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.h2gate = nn.Linear(h_dim, x_dim)
        self.h2std = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        h = self.z2h(z)
        ld = self.h2ld(h)
        ld = self.softplus(ld)
        gate_logits = self.h2gate(h)
        return ld, gate_logits
        

class Decoder_a(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super(Decoder_a, self).__init__()
        self.z2h = LinearGELU(z_dim, h_dim)
        self.h2ld = nn.Linear(h_dim, x_dim)
        self.h2gate = nn.Linear(h_dim, x_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        h = self.z2h(z)
        ld = self.softplus(self.h2ld(h))
        gate_logit = self.h2gate(h)
        return ld, gate_logit


class Encoder_l(nn.Module):
    def __init__(self, x_dim, sh_dim=100):
        super(Encoder_l, self).__init__()
        self.x2h = LinearGELU(x_dim, sh_dim)
        self.h2mu = nn.Linear(sh_dim, 1)
        self.h2logvar = nn.Linear(sh_dim, 1)

    def forward(self, x):
        h = self.x2h(x)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        return mu, logvar

class Encoder_d(nn.Module):
    def __init__(self, z_dim, d_h_dim):
        super(Encoder_d, self).__init__()
        self.z2h = LinearGELU(z_dim, d_h_dim)
        self.h2h = LinearGELU(d_h_dim, d_h_dim)
        self.h2mu = nn.Linear(d_h_dim, z_dim)
        self.h2logvar = nn.Linear(d_h_dim, z_dim)

    def forward(self, x):
        h = self.z2h(x)
        h = self.h2h(h)
        mu = self.h2mu(h)
        logvar = self.h2logvar(h)
        return mu, logvar

class VAE_RNA(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim, d_h_dim, l_prior_r, d_coeff=1e-2):
        super(VAE_RNA, self).__init__()
        self.enc_z = Encoder_su(x_dim, h1_dim, h2_dim, z_dim)
        self.enc_l = Encoder_l(x_dim, sh_dim=100)
        self.enc_d = Encoder_d(z_dim, d_h_dim)
        self.dec_s = Decoder_r(z_dim, h1_dim, x_dim)
        self.dec_u = Decoder_r(z_dim, h1_dim, x_dim)
        self.l_prior_r = l_prior_r
        self.d_coeff = d_coeff
        self.eps = 1e-8

    def forward(self, s, u):
        qz_x = self.enc_su(s, u)
        ql = self.enc_size(s, u)
        z_i, l_i = qz_x.rsample(), ql.rsample()
        s_raw, u_raw = self.dec_su(z_i)
        return z_i, l_i, qz_x, ql, s_raw, u_raw

    def forward_d(self, s, u, z_sample=True):
        qz_x = self.enc_su(s, u)
        ql = self.enc_size(s, u)
        if z_sample:
            z_i, l_i = qz_x.rsample(), ql.rsample()
        else:
            z_i, l_i = qz_x.mean, ql.mean

        qd = self.enc_dyn(z_i)
        d_i = qd.rsample()
        s_raw, u_raw = self.dec_su(z_i)
        s_raw_dt, u_raw_dt = self.dec_su(z_i + d_i * self.d_coeff)
        return z_i, l_i, d_i, qz_x, ql, qd, s_raw, u_raw, s_raw_dt, u_raw_dt

    def enc_su(self, s, u):
        qz_mu, qz_logvar = self.enc_z(s, u)
        qz_x = dist.Normal(qz_mu, F.softplus(qz_logvar)+self.eps)
        return qz_x

    def enc_dyn(self, z):
        qd_mu, qd_logvar = self.enc_d(z)
        qd = dist.Normal(qd_mu, F.softplus(qd_logvar)+self.eps)
        return qd

    def enc_size(self, s, u):
        ql_mu, ql_logvar = self.enc_l(s+u)
        ql = dist.log_normal.LogNormal(ql_mu, F.softplus(ql_logvar)+self.eps)
        return ql

    def dec_su(self, z):
        s_raw, u_raw = self.dec_s(z), self.dec_u(z)
        return s_raw, u_raw


class VAE_ATAC(nn.Module):
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim, l_prior_a):
        super(VAE_ATAC, self).__init__()
        self.enc_z = Encoder_a(x_dim, h1_dim, h2_dim, z_dim)
        self.enc_l = Encoder_l(x_dim, sh_dim=100)
        self.dec_a = Decoder_a(z_dim, h1_dim, x_dim)
        self.l_prior_a = l_prior_a
        self.eps = 1e-8

    def forward(self, x, z_sample=True):
        qz_x = self.enc_a(x)
        ql = self.enc_size(x)
        if z_sample:
            z_i, l_i = qz_x.rsample(), ql.rsample()
        else:
            z_i, l_i = qz_x.mean, ql.mean
        a_params = self.dec_ald(z_i)
        return(z_i, l_i, qz_x, ql, a_params)

    def enc_a(self, x):
        qz_mu, qz_logvar = self.enc_z(x)
        qz_x = dist.Normal(qz_mu, F.softplus(qz_logvar)+self.eps)
        return qz_x

    def enc_size(self, x):
        ql_mu, ql_logvar = self.enc_l(x)
        ql = dist.log_normal.LogNormal(ql_mu, F.softplus(ql_logvar)+self.eps)
        return ql

    def dec_ald(self, z):
        a_raw, gate = self.dec_a(z)
        a_params = [a_raw, gate]
        return a_params