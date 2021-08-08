import torch.nn.functional as F
from torch import nn

criterion_MSE = nn.MSELoss(reduction="mean")


def cross_entropy(y, labels, reduce=True):
    l_ce = F.cross_entropy(y, labels, reduce=reduce)
    return l_ce


def distillation(student_scores, teacher_scores, T):

    p = F.log_softmax(student_scores / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / student_scores.shape[0]

    return l_kl
