import numpy as np
import random
from scipy.stats import norm
from scipy.special import log_expit as log_sigmoid
from scipy.special import expit as sigmoid
from scipy.optimize import minimize, brute
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm, lines

plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', weight='normal', size=18)
# plt.rcParams['mathtext.fontset'] = 'Times New Roman'

# *original data plots (setup_[ab].pdf) produced using NUM_SAMPLES = 100_000
NUM_SAMPLES = 10_000                # How many samples to take from each environment to train/test on
SEED = 0

CMAP_SOLID = cm.get_cmap("tab10")
CMAP_GRADIENT = cm.get_cmap("jet")
# CMAP_GRADIENT = cm.get_cmap("viridis")

COLOR_ENVS = [CMAP_SOLID(0), CMAP_SOLID(1), CMAP_SOLID(2)]
COLOR_Y0, COLOR_Y1 = CMAP_GRADIENT(0.1), CMAP_GRADIENT(0.9)

PLOT_MAX_POINTS = 100
PLOT_MARKER_SIZE = 5
PLOT_MARKER_ALPHA = 0.5
PLOT_MARKER_STYLE = 'o'
PLOT_MARKER_STYLE_Y0 = '^'
PLOT_MARKER_STYLE_Y1 = 's'
PLOT_MIN_R, PLOT_MAX_R = -6, 6
PLOT_MIN_S, PLOT_MAX_S = -6, 6

PLOT_DATA_ONE_ROW = True         # plot single row data plot (joint and conditional data densities only) if false then plot 4 rows (16 plots)

SHOW_DATA_PLOT = True
SHOW_PREDICTORS_PLOT = True
SHOW_ACC_PLOT = False

# set *_PDF variables to empty string or False to prevent saving
DATA_PDF = 'environments.pdf'
PREDICTORS_PDF = 'classifiers.pdf'
ACC_PDF = ''  # 'test_acc.pdf'


class Environment:
    """
    Environment class encapsulates distributional parameters that change between environments
    """
    def __init__(self, mean_r, std_r, mean_s0, std_s0, mean_s1, std_s1):
        self.mean_r = mean_r
        self.std_r = std_r
        self.mean_s0 = mean_s0
        self.std_s0 = std_s0
        self.mean_s1 = mean_s1
        self.std_s1 = std_s1

    def sample_R(self, N):
        return np.random.normal(self.mean_r, self.std_r, N)

    def sample_S_cond_Y(self, y):
        N = y.shape
        s0 = np.random.normal(self.mean_s0, self.std_s0, N)
        s1 = np.random.normal(self.mean_s1, self.std_s1, N)
        return np.where(y, s1, s0)

    def logpdf_R(self, r):
        return norm.logpdf(r, loc=self.mean_r, scale=self.std_r)

    def logpdf_S_cond_Y(self, s, y):
        log_p_s_cond_y0 = norm.logpdf(s, loc=self.mean_s0, scale=self.std_s0)
        log_p_s_cond_y1 = norm.logpdf(s, loc=self.mean_s1, scale=self.std_s1)
        return np.where(y, log_p_s_cond_y1, log_p_s_cond_y0)


class Invariant:
    """
    Invariant class encapsulates distributional parameters that do not change between environments
    """
    def __init__(self, mean_y, std_y):
        self.mean_y = mean_y
        self.std_y = std_y

    def sample_Y_cond_R(self, r):
        N = r.shape[0]
        eps = np.random.normal(self.mean_y, self.std_y, N)
        return (r + eps > 0).astype(int)

    def logpmf_Y_cond_R(self, y, r):
        # P[Y=0 | R=r] = P[r + eps <= 0] = P[-r - eps >= 0] = P[eps <= -r]
        # P[Y=1 | R=r] = P[eps > -r] = 1 - P[eps <= -r]
        log_p_y0_cond_r = norm.logcdf(-r)
        log_p_y1_cond_r = norm.logsf(-r)        # survival function sf(x) = 1 - cdf(x)
        return np.where(y, log_p_y1_cond_r, log_p_y0_cond_r)


class DataModel:
    """
    DataModel provides mechanism to sample data from each environment and provides true pdfs and pmfs
    """
    def __init__(self, environments: List[Environment], invariant: Invariant):
        self.environments = environments
        self.invariant = invariant

        self._p_y0 = []
        self._p_y1 = []
        self._init_prob_y()

    def sample(self, e, N):
        env = self.environments[e]
        r = env.sample_R(N)
        y = self.invariant.sample_Y_cond_R(r)
        s = env.sample_S_cond_Y(y)
        return y, r, s

    def pdf_r_s_y(self, e, r, s, y):
        env = self.environments[e]
        logpdf_r = env.logpdf_R(r)
        logpmf_y_cond_r = self.invariant.logpmf_Y_cond_R(y, r)
        logpdf_s_cond_y = env.logpdf_S_cond_Y(s, y)
        # assumes S is conditionally independent of R given Y (I think this is true based on experiments)
        return np.exp(logpdf_r + logpmf_y_cond_r + logpdf_s_cond_y)

    def pdf_r_s(self, e, r, s):
        p_r_s_y0 = self.pdf_r_s_y(e, r, s, np.zeros(r.shape[-1], dtype=int))
        p_r_s_y1 = self.pdf_r_s_y(e, r, s, np.ones(r.shape[-1], dtype=int))
        return p_r_s_y0 + p_r_s_y1

    def pmf_y(self, e, y):
        return np.choose(y, [self._p_y0[e], self._p_y1[e]])

    def _init_prob_y(self):
        # compute P(Y=0) at initialization using numerical approximation
        for env in self.environments:
            min_r = env.mean_r + self.invariant.mean_y - 6 * (env.std_r + self.invariant.std_y)
            max_r = env.mean_r + self.invariant.mean_y + 6 * (env.std_r + self.invariant.std_y)
            r = np.linspace(min_r, max_r, 10000)
            log_p_y0_cond_r = self.invariant.logpmf_Y_cond_R(np.zeros(r.shape, dtype=int), r)
            log_p_r = env.logpdf_R(r)
            pdf_r_y0 = np.exp(log_p_y0_cond_r + log_p_r)
            self._p_y0.append(np.clip(np.trapz(pdf_r_y0, r), 0, 1))
            self._p_y1.append(1 - self._p_y0[-1])

    def __repr__(self):
        s = f"{self.__class__.__name__}(environments=[\n"
        for e in self.environments:
            s += f"        {e.__class__.__name__}(\n"
            s += f"            mean_r={e.mean_r},\n"
            s += f"            std_r={e.std_r},\n"
            s += f"            mean_s0={e.mean_s0},\n"
            s += f"            std_s0={e.std_s0},\n"
            s += f"            mean_s1={e.mean_s1},\n"
            s += f"            std_s1={e.std_s1}),\n"
        s += "    ],\n"
        s += f"    invariant={self.invariant.__class__.__name__}(\n"
        s += f"        mean_y={self.invariant.mean_y},\n"
        s += f"        std_y={self.invariant.std_y}))\n"
        return s


def plot_model(data_model, one_row=False):
    """
    Plot the data model (no samples are plotted here, just the information about the distributions)
    Assumes there are 3 environments, if one_row=True then plot only 4 plot only P(R,S) and P(R,S|Y).
    """
    LINE_STYLES = ['--', '-.', '-']

    def plot_pmf_y(ax):
        n_env = len(data_model.environments)
        labels = [f"e{e+1}" for e in range(n_env)]
        p_y0 = [data_model.pmf_y(e, 0) for e in range(n_env)]
        p_y1 = [data_model.pmf_y(e, 1) for e in range(n_env)]

        x = np.arange(len(labels))
        width = 0.35

        ax.set_xticks(x, labels)
        rects_y0 = ax.bar(x - width / 2, p_y0, width, label="$P(Y=0)$", color=COLOR_Y0)
        rects_y1 = ax.bar(x + width / 2, p_y1, width, label="$P(Y=1)$", color=COLOR_Y1)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + 0.35)

        ax.set_title("$P(Y)$")
        ax.legend(loc='upper right')

        ax.bar_label(rects_y0, padding=3, fmt="%.2f")
        ax.bar_label(rects_y1, padding=3, fmt="%.2f")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)

    def plot_pmf_y_cond_r(ax):
        r = np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100)
        p_y1_cond_r = np.exp(data_model.invariant.logpmf_Y_cond_R(np.ones(100, dtype=int), r))
        ax.plot(r, p_y1_cond_r, color=CMAP_SOLID(0))
        ax.set_title("$P(Y=1 | R=r)$")
        ax.set_xlabel("$r$")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)

    def plot_pdf_r_s(ax):
        rr, ss = np.meshgrid(np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100), np.linspace(PLOT_MIN_S, PLOT_MAX_S, 100))
        legend_lines = []
        legend_titles = []
        for e in range(len(data_model.environments)):
            p_rr_ss = data_model.pdf_r_s(e, rr.reshape(-1), ss.reshape(-1)).reshape(rr.shape)
            ax.contour(rr, ss, p_rr_ss, levels=5, colors=[COLOR_ENVS[e]], linestyles=LINE_STYLES[e % len(LINE_STYLES)])
            legend_lines.append(lines.Line2D([0], [0], linestyle=LINE_STYLES[e % len(LINE_STYLES)], color=COLOR_ENVS[e]))
            legend_titles.append(f"$p_{e + 1}(x)$")
        # ax.legend(legend_lines, legend_titles, loc='upper right')
        ax.set_xlabel("$x_{inv}$")
        ax.set_ylabel("$x_{spu}$")
        ax.set_title("Data distributions")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)
        return legend_lines, legend_titles

    def plot_pdf_r_s_cond_y(ax, e):
        r = np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100)
        s = np.linspace(PLOT_MIN_S, PLOT_MAX_S, 100)
        rr, ss = np.meshgrid(r, s)
        rr, ss = rr.reshape(-1), ss.reshape(-1)
        y0 = np.zeros(rr.shape, dtype=int)
        y1 = np.ones(rr.shape, dtype=int)
        p_rr_ss_cond_y0 = data_model.pdf_r_s_y(e, rr, ss, y0) / data_model.pmf_y(e, y0)
        p_rr_ss_cond_y1 = data_model.pdf_r_s_y(e, rr, ss, y1) / data_model.pmf_y(e, y1)
        rr, ss, p_rr_ss_cond_y0, p_rr_ss_cond_y1 = rr.reshape(100, 100), ss.reshape(100, 100), \
                                                   p_rr_ss_cond_y0.reshape(100, 100), p_rr_ss_cond_y1.reshape(100, 100)
        ax.contour(rr, ss, p_rr_ss_cond_y0, levels=5, colors=[COLOR_Y0], linestyles=LINE_STYLES[2])
        ax.contour(rr, ss, p_rr_ss_cond_y1, levels=5, colors=[COLOR_Y1], linestyles=LINE_STYLES[1])
        # color_y0_alp = list(COLOR_Y0)
        # color_y0_alp[3] = PLOT_MARKER_ALPHA
        # color_y1_alp = list(COLOR_Y1)
        # color_y1_alp[3] = PLOT_MARKER_ALPHA
        # ax.legend([lines.Line2D([0], [0], color=COLOR_Y0, linestyle=LINE_STYLES[0], marker=PLOT_MARKER_STYLE_Y0,
        #                         markersize=PLOT_MARKER_SIZE, markeredgecolor=color_y0_alp, markerfacecolor=color_y0_alp),
        #            lines.Line2D([0], [0], color=COLOR_Y1, linestyle=LINE_STYLES[1], marker=PLOT_MARKER_STYLE_Y1,
        #                         markersize=PLOT_MARKER_SIZE, markeredgecolor=color_y1_alp, markerfacecolor=color_y1_alp)],
        #           [f"$P(R^{{e{e+1}}}=r, S^{{e{e+1}}}=s | Y=0)$", f"$P(R^{{e{e+1}}}=r, S^{{e{e+1}}}=s | Y=1)$"],
        #           loc='upper right')
        ax.set_xlabel("$x_{inv}$")
        ax.set_ylabel("$x_{spu}$")
        if e < 2:
            ax.set_title(f"Environment {e + 1}")
        else:
            ax.set_title(f"Environment {e + 1} (test)")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)

    def plot_pdf_s_cond_y(ax, e):
        s = np.linspace(PLOT_MIN_S, PLOT_MAX_S, 100)
        y0 = np.zeros(100, dtype=int)
        y1 = np.ones(100, dtype=int)
        ax.plot(np.exp(data_model.environments[e].logpdf_S_cond_Y(s, y0)), s, label=f"$P(S^{{e{e+1}}}=s | Y=0)$",
                color=COLOR_Y0)
        ax.plot(np.exp(data_model.environments[e].logpdf_S_cond_Y(s, y1)), s, label=f"$P(S^{{e{e+1}}}=s | Y=1)$",
                color=COLOR_Y1)
        ax.legend(loc='upper right')
        ax.set_ylabel("$s$")
        ax.set_title(f"$P(S^{{e{e+1}}} | Y)$")
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)

    def plot_pdf_r(ax):
        n_env = len(data_model.environments)
        r = np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100)
        for e in range(n_env):
            ax.plot(np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100), np.exp(data_model.environments[e].logpdf_R(r)),
                    label=f"$P(R^{{e{e+1}}}=r)$", color=COLOR_ENVS[e])
        ax.legend(loc='upper right')
        ax.set_xlabel("$r$")
        ax.set_title("$P(R)$")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)

    def plot_pdf_r_cond_y(ax, e):
        r = np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100)
        y0 = np.zeros(100, dtype=int)
        y1 = np.ones(100, dtype=int)
        # P(R | Y) = P(Y | R) * P(R) / P(Y)
        p_r_cond_y0 = np.exp(data_model.invariant.logpmf_Y_cond_R(y0, r) + data_model.environments[e].logpdf_R(r) -
                             np.log(data_model.pmf_y(e, y0)))
        p_r_cond_y1 = np.exp(data_model.invariant.logpmf_Y_cond_R(y1, r) + data_model.environments[e].logpdf_R(r) -
                             np.log(data_model.pmf_y(e, y1)))
        ax.plot(r, p_r_cond_y0, label=f"$P(R^{{e{e+1}}}=r | Y=0)$", color=COLOR_Y0)
        ax.plot(r, p_r_cond_y1, label=f"$P(R^{{e{e+1}}}=r | Y=1)$", color=COLOR_Y1)
        ax.legend(loc='upper right')
        ax.set_xlabel("$r$")
        ax.set_title(f"$P(R^{{e{e+1}}}|Y)$")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)

    def plot_pmf_y_cond_r_s(ax, e):
        r = np.linspace(PLOT_MIN_R, PLOT_MAX_R, 250)
        s = np.linspace(PLOT_MIN_S, PLOT_MAX_S, 250)
        rr, ss = np.meshgrid(r, s)
        rr, ss = rr.reshape(-1), ss.reshape(-1)
        y0 = np.zeros(rr.shape, dtype=int)
        y1 = np.ones(rr.shape, dtype=int)
        p_r_s_y0 = data_model.pdf_r_s_y(e, rr, ss, y0).reshape(250, 250)
        p_r_s_y1 = data_model.pdf_r_s_y(e, rr, ss, y1).reshape(250, 250)

        p_y1_cond_r_s = p_r_s_y1 / (p_r_s_y0 + p_r_s_y1)
        rr, ss = rr.reshape(250, 250), ss.reshape(250, 250)

        ax.pcolormesh(rr, ss, p_y1_cond_r_s, shading='gouraud', vmin=0, vmax=1, cmap=CMAP_GRADIENT)
        ax.set_title(f"$P(Y^{{e{e+1}}}=1 | R^{{e{e+1}}}=r, S^{{e{e+1}}}=s)$")
        ax.legend(
            [lines.Line2D([0], [0], color=CMAP_GRADIENT(0.0), lw=4),
             lines.Line2D([0], [0], color=CMAP_GRADIENT(0.5), lw=4),
             lines.Line2D([0], [0], color=CMAP_GRADIENT(1.0), lw=4)],
            ["0.0", "0.5", "1.0"],
            loc='upper right')
        ax.set_xlabel("$r$")
        ax.set_ylabel("$s$")
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)

    dpi = 80
    h_pix, w_pix = 280 if one_row else 1400, 1400
    fig: plt.Figure = plt.figure(figsize=(w_pix / dpi, h_pix / dpi * 1.1), dpi=dpi)
    axs: List[List[plt.Axes]] = fig.subplots(1 if one_row else 4, 4, squeeze=False)

    joint_lines, joint_names = plot_pdf_r_s(axs[0][0])
    if not one_row:
        plot_pdf_r(axs[1][0])
        plot_pmf_y_cond_r(axs[2][0])
        plot_pmf_y(axs[3][0])

    plot_pdf_r_s_cond_y(axs[0][1], 0)
    if not one_row:
        plot_pdf_r_cond_y(axs[1][1], 0)
        plot_pdf_s_cond_y(axs[2][1], 0)
        plot_pmf_y_cond_r_s(axs[3][1], 0)

    plot_pdf_r_s_cond_y(axs[0][2], 1)
    if not one_row:
        plot_pdf_r_cond_y(axs[1][2], 1)
        plot_pdf_s_cond_y(axs[2][2], 1)
        plot_pmf_y_cond_r_s(axs[3][2], 1)

    plot_pdf_r_s_cond_y(axs[0][3], 2)
    if not one_row:
        plot_pdf_r_cond_y(axs[1][3], 2)
        plot_pdf_s_cond_y(axs[2][3], 2)
        plot_pmf_y_cond_r_s(axs[3][3], 2)

    color_y0_alp = list(COLOR_Y0)
    color_y0_alp[3] = PLOT_MARKER_ALPHA
    color_y1_alp = list(COLOR_Y1)
    color_y1_alp[3] = PLOT_MARKER_ALPHA
    fig.legend(joint_lines + [
               lines.Line2D([0], [0], color=COLOR_Y0, linestyle=LINE_STYLES[2], marker=PLOT_MARKER_STYLE_Y0,
                            markersize=PLOT_MARKER_SIZE, markeredgecolor=color_y0_alp, markerfacecolor=color_y0_alp),
               lines.Line2D([0], [0], color=COLOR_Y1, linestyle=LINE_STYLES[1], marker=PLOT_MARKER_STYLE_Y1,
                            markersize=PLOT_MARKER_SIZE, markeredgecolor=color_y1_alp, markerfacecolor=color_y1_alp)],
              joint_names + [f"$p(x | Y=0)$", f"$p(x | Y=1)$"],
              loc='right', ncol=1)

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    return fig, axs


def plot_samples(axs, samples_e0, samples_e1, samples_e2):
    """
    Plots samples onto the existing plots returned from plot_model
    """
    y = [samples_e0[0], samples_e1[0], samples_e2[0]]
    r = [samples_e0[1], samples_e1[1], samples_e2[1]]
    s = [samples_e0[2], samples_e1[2], samples_e2[2]]

    def plot_r_s_cond_y(ax, e):
        color_y0 = list(COLOR_Y0)
        color_y0[3] = PLOT_MARKER_ALPHA
        color_y1 = list(COLOR_Y1)
        color_y1[3] = PLOT_MARKER_ALPHA
        r_cond_y0 = r[e][y[e] == 0]
        s_cond_y0 = s[e][y[e] == 0]
        r_cond_y1 = r[e][y[e] == 1]
        s_cond_y1 = s[e][y[e] == 1]
        ax.plot(r_cond_y0[:PLOT_MAX_POINTS], s_cond_y0[:PLOT_MAX_POINTS], marker=PLOT_MARKER_STYLE_Y0, linestyle='None',
                markerfacecolor=color_y0, markeredgecolor=color_y0, markersize=PLOT_MARKER_SIZE)
        ax.plot(r_cond_y1[:PLOT_MAX_POINTS], s_cond_y1[:PLOT_MAX_POINTS], marker=PLOT_MARKER_STYLE_Y1, linestyle='None',
                markerfacecolor=color_y1, markeredgecolor=color_y1, markersize=PLOT_MARKER_SIZE)
        ax.set_xlim(PLOT_MIN_R, PLOT_MAX_R)
        ax.set_ylim(PLOT_MIN_S, PLOT_MAX_S)

    plot_r_s_cond_y(axs[0][1], 0)
    plot_r_s_cond_y(axs[0][2], 1)
    plot_r_s_cond_y(axs[0][3], 2)


def get_data_model():
    """
    Create the data model
    """
    environments = [
        Environment(
            mean_r=0,
            std_r=2,
            mean_s0=1,
            std_s0=0.5,
            mean_s1=-1,
            std_s1=0.5),
        Environment(
            mean_r=0,
            std_r=0.5,
            mean_s0=1,
            std_s0=2,
            mean_s1=-1,
            std_s1=2),
        Environment(
            mean_r=0,
            std_r=3,
            mean_s0=-1,
            std_s0=1,
            mean_s1=1,
            std_s1=1),
    ]

    invariant = Invariant(
        mean_y=0,
        std_y=1)

    data_model = DataModel(environments, invariant)

    return data_model


def generate_data():
    """
    Create a data model and sample it. Return the samples and the data model
    """
    data_model = get_data_model()

    # sample y, r, s from each data model and compute the density of the data on every environment
    retval = []
    n_env = len(data_model.environments)
    for e in range(n_env):
        y, r, s = data_model.sample(e, NUM_SAMPLES)
        retval.append([y, r, s])
        # compute pdf for every sample on every environment
        for ee in range(n_env):
            retval[-1].append(data_model.pdf_r_s(ee, r, s))

    retval.append(data_model)

    return retval


def bce_with_logits_loss(z, y, weight=1, mean_reduce=True):
    loss = -weight * (y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z))
    return np.mean(loss) if mean_reduce else loss


def weighted_softmax(z, y, weight=1):
    """
    returns softmax([0, weight * z])[y] (assuming binary case and z represents logit of y=1)
    """
    # assumes binary classes
    # denominator worked out from
    # exp(weight * 0 * z) = 1
    # exp(weight * 1 * z) = np.exp(weight * z)

    # all three of the following expressions are equal
    # res = sigmoid((2 * y - 1) * weight * z)
    # res = softmax(np.stack([np.zeros_like(z), weight * z]), axis=0)[y, np.arange(z.shape[0])]
    # res = np.exp(weight * y * z) / (1 + np.exp(weight * z))
    return sigmoid((2 * y - 1) * weight * z)


def weighted_log_softmax(z, y, weight=1):
    """
    natural log of weighted_softmax
    """
    return log_sigmoid((2 * y - 1) * weight * z)


def compute_logits(theta, R_e0, S_e0):
    """
    compute logits assuming z = theta . [r,s,1]
    """
    return theta @ np.vstack([R_e0, S_e0, np.ones_like(R_e0)])


def density_weighted_predict(theta, r, s, pdf_e0, pdf_e1):
    """
    compute predicted probability that y=1 using density weighted softmax predictor
    """
    z = compute_logits(theta, r, s)
    weight = (pdf_e0 + pdf_e1) / 2
    predicted_prob_y1 = weighted_softmax(z, np.ones(z.shape, dtype=int), weight)
    return predicted_prob_y1


def erm_loss_fun(theta, samples_e0, samples_e1, *args):
    [y_e0, r_e0, s_e0, pdf_r_s_e0_e0, pdf_r_s_e0_e1, _] = samples_e0
    [y_e1, r_e1, s_e1, pdf_r_s_e1_e0, pdf_r_s_e1_e1, _] = samples_e1

    z_e0 = compute_logits(theta, r_e0, s_e0)
    z_e1 = compute_logits(theta, r_e1, s_e1)

    loss_e0 = bce_with_logits_loss(z_e0, y_e0, mean_reduce=False)
    loss_e1 = bce_with_logits_loss(z_e1, y_e1, mean_reduce=False)
    loss = np.mean(np.concatenate([loss_e0, loss_e1]))

    return loss


def erm(theta0, samples_e0, samples_e1, data_model):
    res = minimize(erm_loss_fun, theta0, args=(samples_e0, samples_e1, data_model))

    print("ERM optimization results:")
    print(res)

    theta = res.x

    def erm_predict(r, s):
        z = compute_logits(theta, r, s)
        return weighted_softmax(z, np.ones(z.shape, dtype=int), 1)

    return erm_predict, np.copy(theta)


def wri_loss_fun(theta, samples_e0, samples_e1, *args):
    [y_e0, r_e0, s_e0, pdf_r_s_e0_e0, pdf_r_s_e0_e1, _] = samples_e0
    [y_e1, r_e1, s_e1, pdf_r_s_e1_e0, pdf_r_s_e1_e1, _] = samples_e1

    z_e0 = compute_logits(theta, r_e0, s_e0)
    z_e1 = compute_logits(theta, r_e1, s_e1)

    loss_e0 = bce_with_logits_loss(z_e0, y_e0, mean_reduce=False)
    loss_e1 = bce_with_logits_loss(z_e1, y_e1, mean_reduce=False)
    loss = np.mean(np.concatenate([loss_e0, loss_e1]))

    wri_loss = (np.mean(loss_e0 * pdf_r_s_e0_e1) - np.mean(loss_e1 * pdf_r_s_e1_e0))**2 / (0.25 * (np.mean(pdf_r_s_e1_e0) + np.mean(pdf_r_s_e0_e1)) * (np.mean(loss_e0) + np.mean(loss_e1)))

    return loss + 1500 * wri_loss


def wri(theta0, samples_e0, samples_e1, data_model):
    res = minimize(wri_loss_fun, theta0, args=(samples_e0, samples_e1, data_model))

    print("WRI optimization results:")
    print(res)

    theta = res.x

    def wri_predict(r, s):
        z = compute_logits(theta, r, s)
        return weighted_softmax(z, np.ones(z.shape, dtype=int), 1)

    return wri_predict, np.copy(theta)


def vrex_loss_fun(theta, samples_e0, samples_e1, *args):
    [y_e0, r_e0, s_e0, pdf_r_s_e0_e0, pdf_r_s_e0_e1, _] = samples_e0
    [y_e1, r_e1, s_e1, pdf_r_s_e1_e0, pdf_r_s_e1_e1, _] = samples_e1

    z_e0 = compute_logits(theta, r_e0, s_e0)
    z_e1 = compute_logits(theta, r_e1, s_e1)

    loss_e0 = bce_with_logits_loss(z_e0, y_e0, mean_reduce=False)
    loss_e1 = bce_with_logits_loss(z_e1, y_e1, mean_reduce=False)
    erm_loss = np.mean(np.concatenate([loss_e0, loss_e1]))

    rex_loss = (np.mean(loss_e0) - np.mean(loss_e1))**2

    return erm_loss + 1 * rex_loss


def vrex(theta0, samples_e0, samples_e1, data_model):
    # can have a hard time converging, do a grid search first to find a good starting point
    # theta0, fval, grid, Jout = brute(
    #     vrex_loss_fun, ((-100, 100), (-100, 100), (-100, 100)), Ns=10, args=(samples_e0, samples_e1, data_model),
    #     disp=True, workers=1, full_output=True)
    res = minimize(vrex_loss_fun, theta0, args=(samples_e0, samples_e1, data_model))

    print("VREx optimization results:")
    print(res)

    theta = res.x

    def vrex_predict(r, s):
        z = compute_logits(theta, r, s)
        return weighted_softmax(z, np.ones(z.shape, dtype=int), 1)

    return vrex_predict, np.copy(theta)


def irm_loss_fun(theta, samples_e0, samples_e1, *args):
    """ ERM loss with IRM regularization """
    [y_e0, r_e0, s_e0, pdf_r_s_e0_e0, pdf_r_s_e0_e1, _] = samples_e0
    [y_e1, r_e1, s_e1, pdf_r_s_e1_e0, pdf_r_s_e1_e1, _] = samples_e1

    z_e0 = compute_logits(theta, r_e0, s_e0)
    z_e1 = compute_logits(theta, r_e1, s_e1)

    loss_e0 = bce_with_logits_loss(z_e0, y_e0, mean_reduce=False)
    loss_e1 = bce_with_logits_loss(z_e1, y_e1, mean_reduce=False)
    erm_loss = np.mean(np.concatenate([loss_e0, loss_e1]))

    # penalty term is the gradient of bce_with_logits_loss(z_e0 * gamma, y_e0) w.r.t. gamma then substitute gamma=1 following IRM code
    irm_penalty_e0 = np.mean(z_e0 * np.where(y_e0, -sigmoid(-z_e0), sigmoid(z_e0)))**2
    irm_penalty_e1 = np.mean(z_e1 * np.where(y_e1, -sigmoid(-z_e1), sigmoid(z_e1)))**2
    irm_loss = (irm_penalty_e0 + irm_penalty_e1) / 2

    # print('ERM IRM:', erm_loss, irm_loss)

    return erm_loss + 10000 * irm_loss


def irm(theta0, samples_e0, samples_e1, data_model):
    # can have a hard time converging, do a grid search first to find a good starting point
    # theta0, fval, grid, Jout = brute(
    #     irm_loss_fun, ((-10, 10), (-10, 10), (-10, 10)), Ns=10, args=(samples_e0, samples_e1, data_model),
    #     disp=True, workers=1, full_output=True)
    res = minimize(irm_loss_fun, theta0, args=(samples_e0, samples_e1, data_model), )

    print("ERM IRM optimization results:")
    print(res)

    theta = res.x

    def irm_predict(r, s):
        z = compute_logits(theta, r, s)
        return weighted_softmax(z, np.ones(z.shape, dtype=int), 1)

    return irm_predict, np.copy(theta)


def plot_predictor(ax, predictor, title):
    """ Plot the predicted probability that Y=1 over data domain (R,S) """
    r, s = np.meshgrid(np.linspace(PLOT_MIN_R, PLOT_MAX_R, 100), np.linspace(PLOT_MIN_S, PLOT_MAX_S, 100))
    pred_prob_y1 = predictor(r.reshape(-1), s.reshape(-1)).reshape(r.shape)
    im = ax.pcolormesh(r, s, pred_prob_y1, cmap=CMAP_GRADIENT, shading='gouraud', vmin=0, vmax=1, rasterized=True)
    ax.set_xlabel("$x_{inv}$")
    ax.set_ylabel("$x_{spu}$")
    ax.set_title(title)
    return im


def plot_predictors(predictors, names):
    """ Plot all the predictors in a single figure with their names as axis titles """
    rows = 1 + (len(names) - 1) // 4
    cols = min(4, len(names))

    fig_pred: plt.Figure = plt.figure(figsize=(1400/80, 280/80 * 1.1), dpi=80)
    axs: List[List[plt.Axes]] = fig_pred.subplots(rows, cols, squeeze=False)
    axs: List[plt.Axes] = [ax for axr in axs for ax in axr]

    used_axs = []
    for pred, name in zip(predictors, names):
        im = plot_predictor(axs[0], pred, name)
        axs[0].set_yticks([-5, 0, 5])   # yaxis.set_major_locator(plt.MaxNLocator(3))   # reduce to 3 ticks on vertical
        used_axs.append(axs.pop(0))
    while axs:
        axs[0].set_visible(False)
        axs.pop(0)

    fig_pred.tight_layout(rect=[0.01, 0, 0.88, 1])

    cbar_ax = fig_pred.add_axes([0.91, 0.1, 0.02, 0.8])
    cbar = fig_pred.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label("$f(x)$", rotation=0, labelpad=25)

    return used_axs, fig_pred


def test_predictors(samples, *predictors):
    """
    Test the accuracy of each predictor on the provided samples
    """
    [y, r, s, _, _, _] = samples
    results = []
    for predictor in predictors:
        pred_prob_y1 = predictor(r, s)
        acc = np.mean((pred_prob_y1 > 0.5).astype(int) == y)
        results.append(acc)
    return results


def plot_test_results(test_results, names):
    """
    Plot test results for each predictor (assumes test_results computed using test_predictors)
    """
    assert len(test_results) == len(names)

    fig_test: plt.Figure = plt.figure(figsize=(5, 5), dpi=80)
    ax_acc = fig_test.subplots(1)
    acc = test_results

    x = np.arange(len(names))
    width = 0.35
    ax_acc.set_xticks(x, names, rotation=45, ha='right')
    rects_acc = ax_acc.bar(x, np.array(acc) * 100, width, color=CMAP_SOLID(0))
    ax_acc.set_ylim(0, 100.0)

    ax_acc.set_title("Test Accuracy")
    ax_acc.bar_label(rects_acc, padding=3, fmt="%.1f%%")

    fig_test.tight_layout()
    return fig_test


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    # create data
    samples_e0, samples_e1, samples_e2, data_model = generate_data()
    print(data_model)

    # plot distributions
    fig_data, axs = plot_model(data_model, one_row=PLOT_DATA_ONE_ROW)
    plot_samples(axs, samples_e0, samples_e1, samples_e2)

    # initial weights
    theta0 = np.random.randn(3)

    experiments = [
        ('ERM', erm),
        ('IRM', irm),
        ('VREx', vrex),
        ('WRI', wri),
    ]

    # run each experiment
    predictors = []
    names = []
    for exp_name, solver in experiments:
        print(f'Solving {exp_name}')
        predict, theta = solver(theta0, samples_e0, samples_e1, data_model)
        predictors.append(predict)
        names.append(exp_name)
        theta0 = theta

    # plot decision function for each predictor
    _, fig_pred = plot_predictors(predictors, names)

    # test each predictor on the test environment
    test_results = test_predictors(
        samples_e2,
        *predictors)

    print("Test Accuracy:")
    for exp_name, acc in zip(names, test_results):
        print(f"    {exp_name}: {acc * 100:0.2f}%")

    # bar graph containing test accuracies
    fig_test = plot_test_results(test_results, names)

    # show plots
    if SHOW_DATA_PLOT:
        fig_data.show()
    if SHOW_ACC_PLOT:
        fig_test.show()
    if SHOW_PREDICTORS_PLOT:
        fig_pred.show()
    if SHOW_DATA_PLOT or SHOW_ACC_PLOT or SHOW_PREDICTORS_PLOT:
        plt.show()

    # save plots
    if PREDICTORS_PDF:
        print(f"Saving predictions plot to {PREDICTORS_PDF}")
        fig_pred.savefig(PREDICTORS_PDF)
    if ACC_PDF:
        print(f"Saving accuracy plot to {ACC_PDF}")
        fig_test.savefig(ACC_PDF)
    if DATA_PDF:
        print(f"Saving data plot to {DATA_PDF}")
        fig_data.savefig(DATA_PDF)


if __name__ == "__main__":
    main()
