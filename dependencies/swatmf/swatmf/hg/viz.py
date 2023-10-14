import matplotlib.pyplot as plt
import numpy as np
from hydroeval import evaluator, nse, rmse, pbias

class Viz(object):

    def __init__(self):
        super().__init__()

    def barToOne(self, df, obds, xlen, ylen, width=None, dates=None, xlabel=None, ylabel=None, output=None):
        if dates is None:
            dates_df = df
        else:
            dates_df = df.loc[dates]
        if width is None:
            width = 0.5
        f, ax = plt.subplots(figsize=(7,6))
        # plot. Set color of marker edge
        flierprops = dict(
                        marker='o', markerfacecolor='#1f77b4', markersize=7,               
                        alpha=0.3)
        ax.boxplot(dates_df.values,positions=obds,
                widths=width,
                flierprops=flierprops,
                showfliers=True,
                showmeans=True,
                meanprops={"marker":"x","markerfacecolor":"white", "markeredgecolor":"blue"}
                )
        ax.plot([xlen[0], xlen[1]], [xlen[0], xlen[1]], '--', color="gray", lw=2, alpha=0.5)
        ax.set_xlim(xlen[0], xlen[1])
        ax.set_ylim(ylen[0], ylen[1])

        y_cal = dates_df.mean()
        corrl_matrix_cal = np.corrcoef(obds, y_cal)
        corrl_xy_cal = corrl_matrix_cal[0,1]
        r_squared_cal = corrl_xy_cal**2
        pbias_cal = evaluator(pbias, np.array(y_cal), np.array(obds))
        ml_cal, b_cal = np.polyfit(obds, y_cal, 1)
        ax.plot(np.array(obds), (ml_cal*np.array(obds)) + b_cal, 'r', alpha=0.5)
        if xlabel is None:
            xlabel = "Observed"
        if ylabel is None:
            ylabel = "Simulated"

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.text(
                0.05, 0.9,
                '$R^2$: {:.3f} | PBIAS: {:.3f}'.format(r_squared_cal, pbias_cal[0]),
                fontsize=18,
                horizontalalignment='left',
                bbox=dict(facecolor='lightgreen'),
                transform=ax.transAxes
                )
        ax.tick_params(axis='both', labelsize=14)
        plt.xticks([(i) for i in range(0,xlen[1], round(xlen[1]/6))], [str(i) for i in range(0,xlen[1], round(xlen[1]/6))])
        ax.grid('True', alpha=0.5)
        if output is None:
            output = 'output.jpg'
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.show()

    def gwlv(self, df, x, y, xlen=None, ylen=None, xlabel=None, ylabel=None, output=None):
        if xlen is None:
            xlen = [df[x].min(), df[x].max()]
            ylen = [df[y].min(), df[y].max()]

        if xlabel is None:
            xlabel = "Observed"
        if ylabel is None:
            ylabel = "Simulated"
        if output is None:
            output = 'gwlv.jpg'      
        groups = df.groupby('grid')
        fig, ax = plt.subplots(figsize=(7,6))
        for name, group in groups:
            ax.scatter(
                    group[x], group[y],
                    s=60,
                    lw=1.5,
                    alpha=0.3,
                    zorder=10,
                    marker='o',
                    label=name)
            
        x_cal = df[x].tolist()
        y_cal = df[y].tolist()
        ax.plot([xlen[0], xlen[1]], [ylen[0], ylen[1]], '--', color="gray", lw=2, alpha=0.5)

        corrl_matrix_cal = np.corrcoef(x_cal, y_cal)
        corrl_xy_cal = corrl_matrix_cal[0,1]
        r_squared_cal = corrl_xy_cal**2
        pbias_cal = evaluator(pbias, df[x].to_numpy(), df[y].to_numpy())
        m_cal, b_cal = np.polyfit(x_cal, y_cal, 1)
        ax.plot(np.array(x_cal), (m_cal*np.array(x_cal)) + b_cal, 'k', alpha=1)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.text(
                0.05, 0.9,
                '$R^2$: {:.3f} | PBIAS: {:.3f}'.format(r_squared_cal, pbias_cal[0]),
                fontsize=18,
                horizontalalignment='left',
                bbox=dict(facecolor='lightgreen'),
                transform=ax.transAxes
                )
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(fontsize=18, loc='lower right')
        plt.savefig(output, dpi=300, bbox_inches="tight")
        ax.grid('True', alpha=0.5)
        plt.show()