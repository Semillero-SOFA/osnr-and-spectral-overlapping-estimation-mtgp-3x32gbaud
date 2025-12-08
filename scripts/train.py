import torch
import gpytorch
from src.utils.data_loader import load_dataset
from src.models.mixed_gp import MixedGPModel
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from pathlib import Path
import json
from datetime import datetime


def train(epochs=50, batch_size=1024, lr=0.01, smoke_test=False):
    print("Loading processed data...")
    # Load data
    train_x, train_y, test_x, test_y, scaler_x, scaler_y = load_dataset()

    if smoke_test:
        print(f"Smoke test: Using subset of data.")
        train_x = train_x[:100]
        train_y = train_y[:100]
        epochs = 1

    print(f"Train size: {train_x.shape}")

    # Create DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    # Inducing points: Use subset (M=1200) for speed optimization.
    # M=1200 is sufficient for N=5000 (25% data) and provides massive speedup over M=N.
    num_inducing = 1200
    if train_x.size(0) < num_inducing:
        num_inducing = train_x.size(0)

    num_latents = 3  # Number of latent functions for LMC

    # Randomly select inducing points
    inducing_idx = torch.randperm(train_x.size(0))[:num_inducing]
    inducing_points = train_x[inducing_idx].clone()

    model = MixedGPModel(inducing_points, num_latents=num_latents, num_tasks=2)

    # Likelihoods - use LikelihoodList for mixed outputs
    # Task 0: Overlap (Binary) -> BernoulliLikelihood
    # Task 1: OSNR (Continuous) -> GaussianLikelihood
    likelihood = gpytorch.likelihoods.LikelihoodList(
        gpytorch.likelihoods.BernoulliLikelihood(),  # Task 0
        gpytorch.likelihoods.GaussianLikelihood(),  # Task 1
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_x, batch_y in pbar:
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            optimizer.zero_grad()

            output = model(batch_x)

            # Manual ELBO computation for LikelihoodList
            # output is MultitaskMultivariateNormal with shape (batch, num_tasks)
            mean = output.mean
            var = output.variance

            # Create distributions for each task
            # Task 0: Overlap (Binary)
            dist_overlap = gpytorch.distributions.MultivariateNormal(
                mean[:, 0], torch.diag_embed(var[:, 0])
            )
            # Task 1: OSNR (Continuous)
            dist_osnr = gpytorch.distributions.MultivariateNormal(
                mean[:, 1], torch.diag_embed(var[:, 1])
            )

            # Expected log probabilities
            num_data = train_x.size(0)
            scale = num_data / batch_x.size(0)

            log_prob_overlap = (
                likelihood.likelihoods[0]
                .expected_log_prob(batch_y[:, 0], dist_overlap)
                .sum()
            )
            log_prob_osnr = (
                likelihood.likelihoods[1]
                .expected_log_prob(batch_y[:, 1], dist_osnr)
                .sum()
            )

            # KL divergence
            kl_div = model.variational_strategy.kl_divergence().sum()

            # ELBO = E[log p(y|f)] - KL
            # We minimize -ELBO
            loss = -(scale * (log_prob_overlap + log_prob_osnr) - kl_div)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_loss:.4f}")

    # Save model with metadata
    checkpoint_dir = Path("checkpoints/processed")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "mixed_gp_model.pth"
    state = {
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),
        "metadata": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_size": train_x.shape[0],
            "test_size": test_x.shape[0],
            "num_inducing": inducing_points.size(0),
            "num_latents": num_latents,
            "timestamp": datetime.now().isoformat(),
        },
    }
    torch.save(state, model_path)
    print(f"Model saved to {model_path}")

    # Save training config
    config_dir = Path("configs/processed")
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(state["metadata"], f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    train(epochs=args.epochs, smoke_test=args.smoke_test)
