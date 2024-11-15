import torch


def sample_zinb(mu, theta, pi):
    """
    Samples from the Zero-Inflated Negative Binomial distribution.

    Parameters
    ----------
    - mu (torch.Tensor): Mean of the NB distribution (batch_size, num_genes)
    - theta (torch.Tensor): Dispersion parameter of the NB distribution (batch_size, num_genes)
    - pi (torch.Tensor): Zero-inflation probability (batch_size, num_genes)

    Returns
    -------
    - samples (torch.Tensor): Sampled counts (batch_size, num_genes)
    """
    # Ensure parameters are on the same device and have the same shape
    assert mu.shape == theta.shape == pi.shape

    # Sample zero-inflation indicator z
    bernoulli_dist = torch.distributions.Bernoulli(probs=pi)
    z = bernoulli_dist.sample()

    # Sample from Negative Binomial distribution
    # Compute probability p from mu and theta
    p = theta / (theta + mu)
    # Convert to total_count (r) and probability (1 - p)
    nb_dist = torch.distributions.NegativeBinomial(total_count=theta, probs=1 - p)
    nb_sample = nb_dist.sample()

    # Combine zero-inflation and NB samples
    samples = z * 0 + (1 - z) * nb_sample

    return samples
