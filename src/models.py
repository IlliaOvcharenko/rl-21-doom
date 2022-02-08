import torch

from efficientnet_pytorch import EfficientNet


class DummyQNet:
    def __init__(self, action_space_size):
        # TODO add stratery like rand, fixed_ation, etc.
        # self.strategy = strategy
        self.action_space_size = action_space_size

    def __call__(self, x):
        batch_size = x.shape[0]
        return torch.rand((batch_size, self.action_space_size))


# TODO implement classical dqn
# class DQN(torch.nn.Module):
    # pass

class DuelQNet(torch.nn.Module):
    def __init__(self, action_space_size, in_channels=3,  encoder_mode="usual"):
        super().__init__()

        assert encoder_mode in ["usual", "effnet"], f"No such encoder option: {encoder_mode}"
        if encoder_mode == "usual":
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, 8, kernel_size=3, stride=2, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),

                torch.nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),

                torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
                torch.nn.BatchNorm2d(8),
                torch.nn.ReLU(),

                torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),

                torch.nn.AdaptiveAvgPool2d((3, 4)),
            )
        elif encoder_mode == "effnet":
            effnet = EfficientNet.from_pretrained("efficientnet-b0", in_channels=in_channels)
            layers_to_remove = ["_fc", "_swish"]
            for l in layers_to_remove:
                setattr(effnet, l, torch.nn.Identity())

            self.encoder = torch.nn.Sequential(
                effnet,
                torch.nn.Linear(1280, 192),
                torch.nn.BatchNorm1d(192),
                torch.nn.ReLU(),
                torch.nn.Linear(192, 192),
            )

        self.state_fc = torch.nn.Sequential(
            torch.nn.Linear(96, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        self.advantage_fc = torch.nn.Sequential(
            torch.nn.Linear(96, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
        return x


if __name__ == "__main__":
    # TODO test models
    # model = DuelQNet(len(actions), 3, "usual")
    model = DuelQNet(8, 12, "effnet")
    x = torch.rand(64, 12, 128, 96).float()
    print(x.dtype)
    out = model(x)
    print(out.shape)

