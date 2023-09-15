import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'ieee', 'bright', 'grid', 'no-latex'])

x = [i for i in range(4)]
x_labels = [20, 40, 60, 80]
y1 = [0.93, 0.92, 0.92, 0.9]
y2 = [0.98, 0.98, 0.96, 0.95]
y3 = [0.66, 0.39, 0.24, 0.19]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.))
param = dict(xlabel='Obstacle Density (%)', ylabel=r'Success rate')

curve_lw = 2.5
marker_size = 6
ax1.set_title("Different obstacle density", fontsize=10)
ax1.plot(x, y3, '-o', lw=curve_lw, markersize=marker_size, label='GLAS')
ax1.plot(x, y2, '--*', lw=curve_lw, markersize=marker_size + 1, label='SBC')
ax1.plot(x, y1, '-^', lw=curve_lw, markersize=marker_size, label='Ours')



ax1.set_xticks(x, x_labels, size='small')
ax1.legend(title='')
ax1.set_ylim(bottom=0.1, top=1.05)
# ax.autoscale(tight=True)
ax1.set(**param)

param = dict(xlabel='Obstacle Size (m)', ylabel=r'Success rate')

x2 = [i for i in range(4)]
x2_labels = [0.6, 0.7, 0.8, 0.85]
y1 = [0.93, 0.91, 0.9, 0.88]
y2 = [0.98, 0.97, 0.72, 0.47]
y3 = [0.19, 0.17, 0.15, 0.14]

ax2.set_title("Different obstacle size", fontsize=10)
ax2.plot(x2, y3, '-o', lw=curve_lw, markersize=marker_size, label='GLAS')
ax2.plot(x2, y2, '--*', lw=curve_lw, markersize=marker_size + 1, label='SBC')
ax2.plot(x2, y1, '-^', lw=curve_lw, markersize=marker_size, label='Ours')

ax2.set_xticks(x2, x2_labels, size='small')
ax2.legend(title='')
ax2.set_ylim(bottom=0.1, top=1.05)
# ax.autoscale(tight=True)
ax2.set(**param)
ax1.tick_params(axis='both', which='both', length=0)
ax2.tick_params(axis='both', which='both', length=0)

# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.grid(color='#B3B3B3', linestyle='-', linewidth=0.25, alpha=0.2)


plt.subplots_adjust(wspace=0.2, hspace=0.25)


fig.savefig('figures/basline_comparison.pdf')
fig.savefig('figures/baseline_comparison.jpg', dpi=300)
