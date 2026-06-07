# copilot offered this:
class ConvSigmaPi2d(nn.Module):
    def __init__(self, channels, kernel_size=3, eps=1e-8):
        super().__init__()
        self.eps = eps

        # Sigma branch (standard conv)
        self.sigma_conv = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size//2
        )

        # Pi branch (log-space conv)
        self.pi_log_weight = nn.Parameter(
            torch.randn(channels, channels, kernel_size, kernel_size) * 0.01
        )
        self.pi_bias = nn.Parameter(torch.zeros(channels))

        # BatchNorm after fusion
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # -------------------------
        # Sigma branch
        # -------------------------
        sigma = self.sigma_conv(x - x.mean(dim=(-2, -1), keepdim=True))

        # -------------------------
        # Pi branch
        # -------------------------
        # Signed log input
        sign_x = torch.sign(x)
        log_mag = torch.log(torch.abs(x) + self.eps)

        # Convolution in log-space
        W = torch.exp(self.pi_log_weight)  # stabilised positive weights
        u = F.conv2d(log_mag, W, bias=self.pi_bias, padding=W.shape[-1]//2)

        # Signed multiplicative output
        sign_out = torch.sign(F.conv2d(sign_x, torch.ones_like(W),
                                       padding=W.shape[-1]//2))
        pi = sign_out * torch.exp(u)

        # -------------------------
        # Geometric mean normalisation
        # -------------------------
        gm = torch.exp(torch.mean(torch.log(torch.abs(pi) + self.eps),
                                  dim=(-2, -1), keepdim=True))
        pi = pi / gm

        # -------------------------
        # Additive fusion
        # -------------------------
        y = sigma + pi

        return F.relu(self.bn(y))


## chatgt rightly made these points:


I think your instinct is good: the implementation in the current paper is more of a signed-log nonlinear branch than a clean Sigma-Pi/product branch. It can still be described as multiplicative-inspired, but I agree that it is not as satisfying as a true product-form layer.

Your proposed version is closer to the mathematical idea,
so the pi branch really is computing a weighted geometric product in log-space. That is a cleaner story.

However, I would be cautious about this exact implementation before committing to rerunning everything.

Main issue: sign handling

This line is not quite right:

sign_out = torch.sign(F.conv2d(sign_x, torch.ones_like(W),
                               padding=W.shape[-1]//2))

A sum of signs is not the sign of a product. For a product, the sign should be the product of signs which flips when there is an odd number of negative inputs. Summing signs and taking sign gives a majority-vote sign, not a product sign.

So if the goal is a true Sigma-Pi branch, I would not use that sign rule.

Second issue: positivity of weights

This line:

W = torch.exp(self.pi_log_weight)

forces all exponents to be positive. That gives you products of positive powers only.

That may be stable, but it removes the ability to learn inverse relationships like:

x1/x2x_1 / x_2x1​/x2​

or suppressive factors. A signed or unconstrained exponent may be more expressive, although harder to stabilise.

A compromise:

W = torch.tanh(self.pi_log_weight)

or

W = 0.1 * torch.tanh(self.pi_log_weight)

This lets exponents be positive or negative but bounded.

Third issue: exp(u) can explode

Even with geometric mean normalisation, this can become numerically aggressive:

pi = sign_out * torch.exp(u)

You probably want to clamp u:

u = torch.clamp(u, -10.0, 10.0)

or use a learnable scale/gate after the pi branch:

pi = torch.exp(self.pi_scale) * pi

with pi_scale initialised negative, as before.

Fourth issue: geometric normalisation changes the interpretation

This part is sensible for stability:

gm = torch.exp(torch.mean(torch.log(torch.abs(pi) + self.eps),
                          dim=(-2, -1), keepdim=True))
pi = pi / gm

But it means the pi branch no longer carries absolute scale, only relative spatial/channel variation. That may be fine, but then the branch needs a learnable amplitude gate afterwards:

pi = torch.exp(self.pi_scale) * pi

Otherwise sigma and pi are being fused without a clear learned relative strength.

What I would implement instead

For a cleaner and trainable version, I would probably use this conceptual form:

class ConvSigmaPi2d(nn.Module):
    def __init__(self, channels, kernel_size=3, eps=1e-6, max_log=6.0):
        super().__init__()
        self.eps = eps
        self.max_log = max_log

        self.sigma_conv = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2
        )

        self.pi_weight_raw = nn.Parameter(
            torch.zeros(channels, channels, kernel_size, kernel_size)
        )
        self.pi_bias = nn.Parameter(torch.zeros(channels))

        self.pi_scale = nn.Parameter(torch.full((channels, 1, 1), -2.0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        sigma = self.sigma_conv(x - x.mean(dim=(-2, -1), keepdim=True))

        # Magnitude product in log-space
        log_mag = torch.log(torch.abs(x) + self.eps)

        # Bounded signed exponents
        W = 0.25 * torch.tanh(self.pi_weight_raw)

        u = F.conv2d(log_mag, W, bias=self.pi_bias, padding=W.shape[-1] // 2)
        u = torch.clamp(u, -self.max_log, self.max_log)

        pi_mag = torch.exp(u)

        # Normalise magnitude to prevent explosion
        gm = torch.exp(
            torch.mean(torch.log(pi_mag + self.eps), dim=(-2, -1), keepdim=True)
        )
        pi_mag = pi_mag / (gm + self.eps)

        # Optional: keep sign separate, or omit sign initially
        pi = torch.exp(self.pi_scale) * pi_mag

        y = sigma + pi
        return F.relu(self.bn(y))

This is not fully signed, but it is much cleaner and more stable. It says:

the pi branch computes bounded weighted geometric products of magnitudes, then learns how strongly to inject them.

If you want signed products, I would treat that as a separate ablation rather than the default.

## copilots reponse

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSigmaPi2d(nn.Module):
    def __init__(self, channels, kernel_size=3,
                 eps=1e-6, max_log=6.0, exponent_scale=0.25):
        super().__init__()
        self.eps = eps
        self.max_log = max_log
        self.exponent_scale = exponent_scale

        # Sigma branch: standard conv
        self.sigma_conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=kernel_size // 2
        )

        # Pi branch: log-space conv over magnitudes
        self.pi_weight_raw = nn.Parameter(
            torch.zeros(channels, channels, kernel_size, kernel_size)
        )
        self.pi_bias = nn.Parameter(torch.zeros(channels))

        # Learnable amplitude for pi branch (starts small)
        self.pi_scale = nn.Parameter(torch.full((channels, 1, 1), -2.0))

        # Fusion normalisation
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # -------------------------
        # Sigma branch
        # -------------------------
        sigma = self.sigma_conv(x - x.mean(dim=(-2, -1), keepdim=True))

        # -------------------------
        # Pi branch (magnitudes only)
        # -------------------------
        log_mag = torch.log(torch.abs(x) + self.eps)

        # Bounded signed exponents
        W = self.exponent_scale * torch.tanh(self.pi_weight_raw)

        u = F.conv2d(log_mag, W, bias=self.pi_bias,
                     padding=W.shape[-1] // 2)
        u = torch.clamp(u, -self.max_log, self.max_log)

        pi_mag = torch.exp(u)

        # Geometric mean normalisation
        gm = torch.exp(
            torch.mean(torch.log(pi_mag + self.eps),
                       dim=(-2, -1), keepdim=True)
        )
        pi_mag = pi_mag / (gm + self.eps)

        # Learnable amplitude
        pi = torch.exp(self.pi_scale) * pi_mag

        # -------------------------
        # Additive fusion
        # -------------------------
        y = sigma + pi
        return F.relu(self.bn(y))
