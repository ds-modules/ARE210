import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cauchy, norm

# Set seaborn style to whitegrid
sns.set_style("whitegrid")

# Define the x-axis range
xval = 13
x = np.linspace(-xval, xval, 1000)

# Compute the CDFs for Cauchy and standard normal distributions
cdf_cauchy = cauchy.cdf(x, loc=0, scale=1)
cdf_normal = norm.cdf(x, loc=0, scale=1)

# Define the quantiles
quantiles = [0.025, 0.50, 0.975]

# Compute the quantile values for Cauchy and standard normal distributions
quantile_values_cauchy = cauchy.ppf(quantiles, loc=0, scale=1)
quantile_values_normal = norm.ppf(quantiles, loc=0, scale=1)

# Concatenate quantile values for extra x-ticks
extra_ticks_x = np.concatenate((quantile_values_cauchy, quantile_values_normal))
extra_ticks_y = quantiles


# Plot the CDFs
plt.plot(x, cdf_cauchy, label='Cauchy', color='red')
plt.plot(x, cdf_normal, label='Standard Normal', color='blue')

# Plot the quantile lines and vertical lines
colors = ['red', 'blue', 'green']
linestyles = ['--', '--', '--']
for i in range(len(quantiles)):
    plt.hlines(quantiles[i], xmin=-xval, xmax=quantile_values_cauchy[i], color=colors[i], linestyle=linestyles[i])
    plt.hlines(quantiles[i], xmin=-xval, xmax=quantile_values_normal[i], color=colors[i], linestyle=linestyles[i])
    plt.plot(quantile_values_cauchy[i], quantiles[i], 'o', color=colors[i])
    plt.plot(quantile_values_normal[i], quantiles[i], 'o', color=colors[i])
    plt.vlines(quantile_values_cauchy[i], ymin=0, ymax=quantiles[i], color=colors[i], linestyle=linestyles[i])
    plt.vlines(quantile_values_normal[i], ymin=0, ymax=quantiles[i], color=colors[i], linestyle=linestyles[i])
    # Find the index of the x-value where the CDF is closest to the quantile value
    idx_cauchy = np.abs(cdf_cauchy - quantiles[i]).argmin()
    idx_normal = np.abs(cdf_normal - quantiles[i]).argmin()

    # Plot a horizontal line from the vertical line to the CDF curve
    plt.plot([quantile_values_cauchy[i], x[idx_cauchy]], [quantiles[i], quantiles[i]], color=colors[i], linestyle=linestyles[i])
    plt.plot([quantile_values_normal[i], x[idx_normal]], [quantiles[i], quantiles[i]], color=colors[i], linestyle=linestyles[i])


# Set the extra ticks for the x-axis
plt.xticks(list(plt.xticks()[0]) + list(extra_ticks_x), color='black')
plt.yticks(list(plt.yticks()[0]) + list(extra_ticks_y), color='red')
    
# Set the y-axis lower limit to 0
plt.ylim(bottom=-0.01)
plt.ylim(top=1.01)
plt.xlim(left=-xval)
plt.xlim(right=xval)
plt.legend()
plt.xlabel('x')
plt.ylabel('CDF')
plt.title('Quantiles Comparisons: Cauchy vs $\mathcal{N}(0,1)$')

plt.tight_layout()  # Adjust spacing
plt.tight_layout()  # Adjust the padding and spacing
plt.savefig('quantile.png', dpi=600)  # Save the figure as tdist.png with higher resolution
plt.show()
