import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rose_plot(data: pd.DataFrame, ax=None, **kwargs):

  if ax is None:
    ax = plt.subplot(polar=True)

  polar_ticks = np.linspace(0., 2 * np.pi, data['task'].nunique() + 1)
  ax.set_xticks(polar_ticks[:-1])
  xticks_labels = [d[:5] for d in data['task'].to_list()]  # shorten task labels
  ax.set_xticklabels(xticks_labels, fontsize=kwargs.get('fontsize', 12))

  # If you want the first axis to be on top:
  ax.set_theta_offset(np.pi / 2)
  ax.set_theta_direction(-1)

  # Draw ylabels
  ax.set_rlabel_position(0)
  ax.set_yticklabels([])

  bar_width = 2 * np.pi / len(data)
  bar_x = [i * bar_width for i in data.index]

  ax.set_ylim(0, data.drop(columns=['task', 'index'], errors='ignore').max().max() + 0.001)

  for col in data.columns:
    if col in ['task', 'index']:
      continue  # skip the task and index columns

    bars = ax.fill(
        bar_x,
        data[col],
        edgecolor='white',
        label=col,
        alpha=kwargs.get('alpha', 0.5))
