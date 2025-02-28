import torch as t
from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import (
    cal_bpr_loss,
    reg_params,
    cal_triplet_loss,
    cal_infonce_loss,
)
from torch import nn

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class MF_my(BaseModel):
    def __init__(self, data_handler):
        super(MF_my, self).__init__(data_handler)

        self.user_int_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size // 2))
        )
        self.item_int_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size // 2))
        )
        self.user_pop_embeds = nn.Parameter(
            init(t.empty(self.user_num, self.embedding_size // 2))
        )
        self.item_pop_embeds = nn.Parameter(
            init(t.empty(self.item_num, self.embedding_size // 2))
        )

        self.is_training = False

        # hyper-parameter
        self.reg_weight = self.hyper_config["reg_weight"]
        self.align_weight = self.hyper_config["align_weight"]
        self.cl_weight = self.hyper_config["cl_weight"]
        self.cl_temperature = self.hyper_config["cl_temperature"]

        dis_loss = configs["model"]["dis_loss"]
        if dis_loss == "L1":
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == "L2":
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == "dcor":
            self.criterion_discrepancy = self.dcor

        self.align_criterion = nn.L1Loss()

        self.int_weight = configs["model"]["int_weight"]
        self.pop_weight = configs["model"]["pop_weight"]
        self.dis_pen = configs["model"]["dis_pen"]

        # semantic-embeddings
        self.usrprf_embeds = (
            t.tensor(configs["usrprf_embeds"]).float().to(configs["device"])
        )
        self.itmprf_embeds = (
            t.tensor(configs["itmprf_embeds"]).float().to(configs["device"])
        )
        self.mlp = nn.Sequential(
            nn.Linear(
                self.usrprf_embeds.shape[1],
                (self.usrprf_embeds.shape[1] + self.embedding_size // 2) // 2,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                (self.usrprf_embeds.shape[1] + self.embedding_size // 2) // 2,
                self.embedding_size // 2,
            ),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    @staticmethod
    def dcor(X1, X2):
        def _create_centered_distance(X):
            """
            Used to calculate the distance matrix of N samples
            """
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            r = t.sum(t.square(X), 1, keepdims=True)
            w = t.bmm(X.unsqueeze(1), X.unsqueeze(-1)).squeeze(-1)
            D = t.sqrt(t.maximum(r - 2 * w + r.t(), t.tensor(0.0)) + 1e-8)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = (
                D
                - t.mean(D, dim=0, keepdims=True)
                - t.mean(D, dim=1, keepdims=True)
                + t.mean(D)
            )

            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = D1.shape[0]
            dcov = t.sqrt(
                t.maximum(t.sum(D1 * D2) / (n_samples * n_samples), t.tensor(0.0))
                + 1e-8
            )
            # dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2)) / n_samples)
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (t.sqrt(t.maximum(dcov_11 * dcov_22, t.tensor(0.0))) + 1e-10)

        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    @staticmethod
    def bpr_loss(p_score, n_score):
        return -t.mean(t.log(t.sigmoid(p_score - n_score)))

    def adapt(self, decay):
        self.int_weight *= decay
        self.pop_weight *= decay
        # self.cl_weight *= decay
        self.align_weight *= decay

    @staticmethod
    def mask_bpr_loss(p_score, n_score, mask):
        return -t.mean(mask * t.log(t.sigmoid(p_score - n_score)))

    @staticmethod
    def _pick_embeds(user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds

    def cal_loss(self, batch_data):
        self.is_training = True
        user, item_p, item_n, mask = batch_data
        mask = mask.bool()
        users_int, items_p_int, items_n_int = self._pick_embeds(
            self.user_int_embeds, self.item_int_embeds, (user, item_p, item_n)
        )
        users_pop, items_p_pop, items_n_pop = self._pick_embeds(
            self.user_pop_embeds, self.item_pop_embeds, (user, item_p, item_n)
        )

        p_score_int = t.sum(users_int * items_p_int, 1)
        n_score_int = t.sum(users_int * items_n_int, 1)
        loss_int = self.mask_bpr_loss(p_score_int, n_score_int, mask)

        p_score_pop = t.sum(users_pop * items_p_pop, 1)
        n_score_pop = t.sum(users_pop * items_n_pop, 1)
        loss_pop = self.mask_bpr_loss(
            n_score_pop, p_score_pop, mask
        ) + self.mask_bpr_loss(p_score_pop, n_score_pop, ~mask)

        p_score_total = p_score_int + p_score_pop
        n_score_total = n_score_int + n_score_pop
        loss_total = self.bpr_loss(p_score_total, n_score_total)

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)

        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(
            usrprf_embeds, itmprf_embeds, (user, item_p, item_n)
        )
        p_score_pre = t.sum(ancprf_embeds * posprf_embeds, 1)
        n_score_pre = t.sum(ancprf_embeds * negprf_embeds, 1)
        loss_pre = self.align_criterion(
            mask * t.sigmoid(p_score_int - n_score_int),
            mask * t.sigmoid(p_score_pre - n_score_pre),
        )
        loss_pre *= self.align_weight

        cl_loss = (
            cal_triplet_loss(users_int, ancprf_embeds, users_pop, self.cl_temperature)
            + cal_triplet_loss(
                items_p_int, posprf_embeds, items_p_pop, self.cl_temperature
            )
            + cal_triplet_loss(
                items_n_int, negprf_embeds, items_n_pop, self.cl_temperature
            )
        )

        cl_loss *= self.cl_weight

        reg_loss = self.reg_weight * reg_params(self)
        losses = {
            "loss_total": loss_total,
            "loss_int": self.int_weight * loss_int,
            "loss_pop": self.pop_weight * loss_pop,
            "reg_loss": reg_loss,
            "cl_loss": cl_loss,
            "loss_pre": loss_pre,
        }
        loss = sum(losses.values())
        return loss, losses

    def full_predict(self, batch_data):
        self.is_training = False
        user_embeds = t.cat((self.user_int_embeds, self.user_pop_embeds), 1)
        item_embeds = t.cat((self.item_int_embeds, self.item_pop_embeds), 1)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
