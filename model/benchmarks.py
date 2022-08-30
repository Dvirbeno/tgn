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
    def get_method_scores(self, team_composition_dict):
        return

    @abstractmethod
    def get_entity_params(self, ids):
        return

    @abstractmethod
    def update_params_by_results(self, composition, player_weights, scores, ranks):
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

    def get_method_scores(self, team_composition_dict):
        player_params_by_team = {team_id: self.get_entity_params(member_players) for team_id, member_players in
                                 team_composition_dict.items()}
        team_agg = {team_id: player_params.mean() for team_id, player_params in player_params_by_team.items()}
        player_weights = {team_id: np.divide(player_params, player_params.sum()) for team_id, player_params in
                          player_params_by_team.items()}

        return {'mu': np.array(list(team_agg.values()))}, {'mu': player_weights}

    def is_entity_known(self, ids):
        return np.array([entity_id in self.mu_per_id for entity_id in ids])

    def update_params_by_results(self, composition, player_weights, scores, ranks):
        scores = scores['mu']
        N = len(scores)
        num_pairwise_comb = comb(N, 2, exact=True)

        mu_diff = np.subtract.outer(scores, scores)
        prob_win_numerator = np.power(1 + np.exp(mu_diff.astype(float) / self.D), -1)
        # zero out diagonal
        np.fill_diagonal(prob_win_numerator, 0.)
        denom = num_pairwise_comb
        prob_win = np.sum(prob_win_numerator, axis=1) / denom

        normalized_rank = (N - ranks) / num_pairwise_comb
        # update "team rating"
        correction_terms = self.k * (normalized_rank - prob_win)
        updated_team_mus = scores + correction_terms

        for (team_id, team_members), (_, weights_per_team), correction in zip(composition.items(),
                                                                              player_weights['mu'].items(),
                                                                              correction_terms):
            for member_id, member_weight in zip(team_members, weights_per_team):
                self.mu_per_id[member_id] += member_weight * correction


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

    def get_method_scores(self, team_composition_dict):
        team_mu, team_sigma, mu_weights, sigma_weights = dict(), dict(), dict(), dict()
        for team_id, member_players in team_composition_dict.items():
            mus, sigmas = self.get_entity_params(member_players)

            team_mu[team_id] = mus.mean()
            team_sigma[team_id] = sigmas.mean()

            mu_weights[team_id] = np.divide(mus, mus.sum())
            sigma_weights[team_id] = np.divide(sigmas, sigmas.sum())

        return {
                   'mu': np.array(list(team_mu.values())),
                   'sigma': np.array(list(team_sigma.values()))}, \
               {
                   'mu': mu_weights, 'sigma': sigma_weights
               }

    def is_entity_known(self, ids):
        return np.array([entity_id in self.mu_per_id for entity_id in ids])

    def update_params_by_results(self, composition, player_weights, scores, ranks):
        mus, sigmas = scores['mu'], scores['sigma']
        N = len(mus)
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

        # compute correction term
        mu_correction_terms = (self.g_operator(sigmas) * (normalized_rank - prob_win) * self.q) / (
                (1 / (sigmas ** 2)) + (1 / d_squared))
        # update "team params"
        updated_mus = mus + mu_correction_terms
        updated_sigmas = np.sqrt(np.power((1 / (sigmas ** 2)) + (1 / d_squared), -1))
        team_sigma_diffs = updated_sigmas - sigmas

        for (team_id, team_members), (_, mu_weights_per_team), (
                _, sigma_weights_per_team), mu_correction, sigma_correction in zip(composition.items(),
                                                                                   player_weights['mu'].items(),
                                                                                   player_weights['sigma'].items(),
                                                                                   mu_correction_terms,
                                                                                   team_sigma_diffs):
            for member_id, mu_member_weight, sigma_member_weight in zip(team_members,
                                                                        mu_weights_per_team,
                                                                        sigma_weights_per_team):
                self.mu_per_id[member_id] += mu_member_weight * mu_correction
                self.sigma_per_id[member_id] += sigma_member_weight * sigma_correction

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

    def get_method_scores(self, team_composition_dict):
        team_scores = []
        for team_id, member_players in team_composition_dict.items():
            team_scores.append(np.array([self.ts_per_id[entity_id].mu for entity_id in member_players]).mean())

        return {'mu': np.array(team_scores)}, None

    def is_entity_known(self, ids):
        return np.array([entity_id in self.ts_per_id for entity_id in ids])

    def update_params_by_results(self, composition, player_weights, scores, ranks):
        if ranks.min() == 1:
            ranks -= 1
        ts_composition = [tuple(self.ts_per_id[entity_id] for entity_id in team_members) for _, team_members in
                          composition.items()]
        updated_skills = self.ts_env.rate(ts_composition, ranks=ranks.tolist())

        for (team_id, team_members), team_skills in zip(composition.items(), updated_skills):
            for member_id, member_skill in zip(team_members, team_skills):
                self.ts_per_id[member_id] = member_skill
