class MonoSceneFeatureAdapter(nn.Module):
    def __init__(self, eps=1e-6, refine=True):
        super().__init__()
        self.eps = eps

        if refine:
            self.refine = nn.Sequential(
                nn.Conv3d(64, 64, kernel_size=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv3d(64, 64, kernel_size=1, bias=False),
            )
        else:
            self.refine = nn.Identity()

    def forward(self, x):
        """
        x: [B, 65, 256, 256, 32]
        return: [B, 64, 128, 128, 16]
        """
        feat = x[:, :64]
        conf = x[:, 64:65].clamp_min(0.0)

        num = F.avg_pool3d(feat * conf, kernel_size=2, stride=2)
        den = F.avg_pool3d(conf, kernel_size=2, stride=2)

        out = num / (den + self.eps)
        out = self.refine(out)
        return out