import torch
import torch.nn.functional as F


class MakeupGANTrainer:
    def __init__(self, G, D, loader, device):
        self.G = G.to(device)
        self.D = D.to(device)
        self.loader = loader
        self.device = device

        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    def train_epoch(self):
        self.G.train()
        self.D.train()

        for batch in self.loader:
            eye = batch["non_makeup"].to(self.device)
            gt = batch["makeup"].to(self.device)
            mask = batch["mask"].to(self.device)
            style = batch["style"].to(self.device)

            # =====================
            # Train Discriminator
            # =====================
            with torch.no_grad():
                fake = self.G(eye, mask, style)

            real_pred = self.D(eye, gt, mask)
            fake_pred = self.D(eye, fake, mask)

            loss_d = (
                F.mse_loss(real_pred, torch.ones_like(real_pred)) +
                F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
            ) * 0.5

            self.opt_d.zero_grad()
            loss_d.backward()
            self.opt_d.step()

            # =====================
            # Train Generator
            # =====================
            fake = self.G(eye, mask, style)
            pred = self.D(eye, fake, mask)

            adv_loss = F.mse_loss(pred, torch.ones_like(pred))
            l1_loss = F.l1_loss(fake * mask, gt * mask)

            loss_g = adv_loss + 10.0 * l1_loss

            self.opt_g.zero_grad()
            loss_g.backward()
            self.opt_g.step()
