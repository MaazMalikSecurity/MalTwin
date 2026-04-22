import torch
from torch.utils.data import WeightedRandomSampler


class ClassAwareOversampler:
    """
    Wraps torch.utils.data.WeightedRandomSampler.
    SRS ref: FE-3 of Module 4
    """

    def __init__(self, dataset, strategy: str = "oversample_minority"):
        self.dataset = dataset
        self.strategy = strategy

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Returns sampler that draws len(dataset) samples per epoch with replacement.
        """
        class_counts = self.dataset.class_counts
        class_names = self.dataset.class_names
        label_map = {name: idx for idx, name in enumerate(class_names)}

        # Build per-class weights
        class_weights: dict[int, float] = {}
        for name, count in class_counts.items():
            idx = label_map[name]
            if count == 0:
                class_weights[idx] = 0.0
            elif self.strategy == "oversample_minority":
                class_weights[idx] = 1.0 / count
            elif self.strategy == "sqrt_inverse":
                import math
                class_weights[idx] = 1.0 / math.sqrt(count)
            elif self.strategy == "uniform":
                class_weights[idx] = 1.0
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Build per-sample weights list
        sample_weights = []
        for _, label in self.dataset._samples:
            sample_weights.append(class_weights.get(label, 0.0))

        weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
        sampler = WeightedRandomSampler(
            weights=weights_tensor,
            num_samples=len(self.dataset),
            replacement=True,
        )
        return sampler
