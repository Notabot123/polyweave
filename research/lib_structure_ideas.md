##naming

name (final): PolyWeave (package `polyweave`) — fits future polynomial/higher-order layers.

## structure

polyweave/
  __init__.py

  ops/
    signed_log.py
    sigma_pi.py
    polynomial.py
    tensor_ops.py

  layers/
    sigmapi_linear.py
    sigmapi_conv.py
    gated.py
    hyperlinear.py

  hypernets/
    base.py
    mlp_hypernet.py
    conv_hypernet.py
    sigmapi_hypernet.py
    heads.py

  targets/
    fc.py
    conv.py
    attention.py
    adapters.py

  prototypes/
    image_stats.py
    feature_stats.py
    sequence_stats.py

  students/
    cnn.py
    transformer.py

  training/
    loops.py
    recovery.py
    warm_restarts.py
    metrics.py

  experiments/
    cifar_fc.py
    cifar_conv1.py
    synthetic_attention.py

  viz/
    plots.py
    diagrams.py

### notes
keep /ops for low-level reusable maths:

signed_log(x)
signed_log1p(x)
sigma_branch(...)
pi_branch(...)
safe_geometric_product(...)

Then /layers wraps those into PyTorch modules:

ConvSigmaPi2d
LinearSigmaPi
GatedSigmaPiBlock

Then /hypernets contains actual weight-generating networks:

ConvFilterHyperNet
FCHyperNet
AttentionQKHyperNet
SigmaPiHyperNet

The important distinction:

ops       = functions
layers    = nn.Module building blocks
hypernets = full networks that generate weights
targets   = packing/unpacking generated parameters

I would definitely include /targets. It will save you pain. A lot of hypernetwork messiness is just:

flat vector → correctly shaped weight dict

So have things like:

FCTargetSpec(num_classes, feature_dim)
Conv2dTargetSpec(out_ch, in_ch, kernel)
AttentionQKTargetSpec(d_model)

Each can expose:

.num_params
.pack(vector)
.unpack(weight_dict)
.install(student, weights)