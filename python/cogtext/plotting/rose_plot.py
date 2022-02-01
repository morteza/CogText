import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rose_plot(data: pd.DataFrame, ax=None, **kwargs):

  if ax is None:
    ax = plt.subplot(polar=True)

  ax.set_rmax(2)

  polar_ticks = np.linspace(0., -2 * np.pi, data['task'].nunique())
  ax.set_xticks(polar_ticks)
  xticks_labels = [d[:4] for d in data['task'].to_list()]  # shorten task labels
  xticks_labels[-1] = ''
  ax.set_xticklabels(xticks_labels)

  label_angles = np.linspace(0, 2 * np.pi, len(xticks_labels)) + (np.pi / 2)
  label_angles[np.cos(label_angles) < 0] = label_angles[np.cos(label_angles) < 0] + np.pi
  label_angles = np.rad2deg(label_angles)

  for label, angle in zip(ax.get_xticklabels(), label_angles):
      x, y = label.get_position()
      lab = ax.text(x, y+.05, label.get_text(), transform=label.get_transform(),
                    fontsize=kwargs.get('fontsize', 8),
                    color='gray',
                    ha=label.get_ha(), va=label.get_va())
      lab.set_rotation(angle)
  ax.set_xticklabels([])

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
        alpha=kwargs.get('alpha', 0.3))
