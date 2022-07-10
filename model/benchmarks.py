from collections import defaultdict
from abc import ABC, abstractmethod
import math
from scipy.special import comb
import numpy as np
from scipy.stats import norm
from trueskill import Rating, rate, TrueSkill


class AbstractSkillMethod(ABC):
    def __init__(self, **default_params):
        self.set_default_parmas(**default_params)
        self.allocate_state_containers()

    def set_default_parmas(self, **default_params):
        for k, v in default_params.items():
            setattr(self, k, v)

    @abstractmethod
    def allocate_state_containers(self):
        return

    @abstractmethod
    def get_method_scores(self, ids):
        return

    @abstractmethod
    def get_entity_params(self, ids):
        return

    @abstractmethod
    def update_params_by_results(self, ids, ranks):
        return

    @abstractmethod
    def is_entity_known(self, ids):
        return


class ELOSkillMethod(AbstractSkillMethod):

    def __init__(self, **default_params):
        self.mu = 1500
        self.k = 10
        self.D = 400
        super(ELOSkillMethod, self).__init__(**default_params)

    def get_default_mu(self):
        return self.mu

    def allocate_state_containers(self):
        self.mu_per_id = defaultdict(self.get_default_mu)

    def get_entity_params(self, ids):
        return np.array([self.mu_per_id[entity_id] for entity_id in ids])

    def get_method_scores(self, ids):
        return self.get_entity_params(ids)

    def is_entity_known(self, ids):
        return np.array([entity_id in self.mu_per_id for entity_id in ids])

    def update_params_by_results(self, ids, ranks):
        mus = self.get_entity_params(ids)
        N = len(ids)
        num_pairwise_comb = comb(N, 2, exact=True)

        mu_diff = np.subtract.outer(mus, mus)
        prob_win_numerator = np.power(1 + np.exp(mu_diff.astype(float) / self.D), -1)
        # zero out diagonal
        np.fill_diagonal(prob_win_numerator, 0.)
        denom = num_pairwise_comb
        prob_win = np.sum(prob_win_numerator, axis=1) / denom

        normalized_rank = (N - ranks) / num_pairwise_comb
        updated_mus = mus + self.k * (normalized_rank - prob_win)

        for entity_id, new_mu in zip(ids, updated_mus):
            self.mu_per_id[entity_id] = new_mu


class GlickoSkillMethod(AbstractSkillMethod):
    def __init__(self, **default_params):
        self.mu = 1500
        self.sigma = 350
        self.q = 0.0057565
        self.D = 400
        super(GlickoSkillMethod, self).__init__(**default_params)

    def get_default_mu(self):
        return self.mu

    def get_default_sigma(self):
        return self.sigma

    def allocate_state_containers(self):
        self.mu_per_id = defaultdict(self.get_default_mu)
        self.sigma_per_id = defaultdict(self.get_default_sigma)

    def get_entity_params(self, ids):
        return np.array([self.mu_per_id[entity_id] for entity_id in ids]), np.array(
            [self.sigma_per_id[entity_id] for entity_id in ids])

    def get_method_scores(self, ids):
        return np.array([self.mu_per_id[entity_id] for entity_id in ids])

    def is_entity_known(self, ids):
        return np.array([entity_id in self.mu_per_id for entity_id in ids])

    def update_params_by_results(self, ids, ranks):
        mus, sigmas = self.get_entity_params(ids)
        N = len(ids)
        num_pairwise_comb = comb(N, 2, exact=True)

        mu_diff = np.subtract.outer(mus, mus)
        sigma_sum = np.sqrt(np.add.outer(sigmas ** 2, sigmas ** 2))
        exponential = -self.g_operator(sigma_sum) * mu_diff / self.D
        prob_win_numerator = np.power(1 + np.power(10, exponential), -1)

        # zero out diagonal
        np.fill_diagonal(prob_win_numerator, 0.)
        denom = num_pairwise_comb
        prob_win = np.sum(prob_win_numerator, axis=1) / denom

        # compute Hessian of log marginal likelihood
        d_squared = np.power((self.q ** 2) * (self.g_operator(sigmas) ** 2) * prob_win * (1 - prob_win), -1)

        normalized_rank = (N - ranks) / num_pairwise_comb
        updated_mus = mus + (self.g_operator(sigmas) * (normalized_rank - prob_win) * self.q) / (
                (1 / (sigmas ** 2)) + (1 / d_squared))
        updated_sigmas = np.sqrt(np.power((1 / (sigmas ** 2)) + (1 / d_squared), -1))

        for entity_id, new_mu, new_sigma in zip(ids, updated_mus, updated_sigmas):
            self.mu_per_id[entity_id] = new_mu
            self.sigma_per_id[entity_id] = new_sigma

    def g_operator(self, sigma):
        return np.power(np.sqrt((1 + 3 * (self.q ** 2) * (sigma ** 2)) / (math.pi ** 2)), -1)


class TrueSkillMethod(AbstractSkillMethod):
    def __init__(self, **default_params):
        self.mu = 25
        self.sigma = 8.333
        self.beta = 4.16
        self.tau = 0.833
        super(TrueSkillMethod, self).__init__(**default_params)

    def allocate_state_containers(self):
        self.ts_env = TrueSkill(mu=self.mu, sigma=self.sigma,
                                beta=self.beta, tau=self.tau, backend='mpmath')
        self.ts_per_id = defaultdict(self.ts_env.create_rating)

    def get_entity_params(self, ids):
        return [self.ts_per_id[entity_id] for entity_id in ids]

    def get_method_scores(self, ids):
        return np.array([self.ts_per_id[entity_id].mu for entity_id in ids])

    def is_entity_known(self, ids):
        return np.array([entity_id in self.ts_per_id for entity_id in ids])

    def update_params_by_results(self, ids, ranks):
        if ranks.min() == 1:
            ranks -= 1

        updated_skills = self.ts_env.rate([(self.ts_per_id[entity_id],) for entity_id in ids],
                                          ranks=ranks.tolist())

        for entity_id, new_skill in zip(ids, updated_skills):
            self.ts_per_id[entity_id] = new_skill[0]
