import torch
import torch.nn as nn
# def rank_loss(features1, features2, margin):
#     cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) * 2.
#     sim = cos(features1, features2)
#     diag = torch.diag(sim)
#     sim = sim - diag.view(-1,1)
#     sim[sim + margin < 0] = 0
#     return sim.mean()
def cross_modal_contrastive_ctriterion(fea, args = None):
        n_view = 2
        batch_size = fea[0].shape[0]
        all_fea = torch.cat(fea)
        sim = all_fea.mm(all_fea.t())
        sim = sim.exp()
        sim = sim - sim.diag().diag()
        sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p1 = diag1 / sim.sum(1)
        loss1 = -(p1).log()

        sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(n_view)])
        diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(n_view)])
        p2 = diag2 / sim.sum(1)
        loss2 = -p2.log()
        return loss1.mean() + loss2.mean()
class SupConLoss(nn.Module):
    def __init__(self, data_class=10, gamma = 3):
        super(SupConLoss, self).__init__()
        self.data_class = data_class
        self.gamma = gamma

    def forward(self, features1, features2, predict1, predict2, labels, epoch_cur, args, img_tau_cur = None, txt_tau_cur = None):
        gamma = self.gamma
        alpha = args.alpha
        clean_idx, hard_idx, noisy_idx = 0, 0, 0
        # divided
        predict = 1 - (1 - predict1) * (1 - predict2)
        predict = (labels * predict).sum(1)
        pass1 = predict > img_tau_cur
        pass2 = predict > txt_tau_cur
        clean_idx = torch.nonzero(pass1 & pass2, as_tuple=False).squeeze()
        hard_idx = torch.nonzero(pass1 ^ pass2, as_tuple=False).squeeze()
        noisy_idx = torch.nonzero(~pass1 & ~pass2, as_tuple=False).squeeze()


        tmp1 = - (labels * predict1.log()).sum(1)
        tmp2 = - (labels * predict2.log()).sum(1)
        term1 = tmp1 + tmp2

        term2 = cross_modal_contrastive_ctriterion([features1, features2], args=args)
        return gamma * term1.mean() + alpha * term2, clean_idx, hard_idx, noisy_idx
        