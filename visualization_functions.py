import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scale_units(value):
    if value < .99:
        new_val = str(round(value,3))
    elif value < 1000:
        new_val = str(round(value))
    elif value < 1000000:
        new_val = str(round(value/1000))+"k"
    elif value < 1 * 10**9:
        new_val = str(round(value/(10**6)))+"M"
    elif value < 1 * 10**12:
        new_val = str(round(value/(10**9)))+"B"
    elif value < 1 * 10**15:
        new_val = str(round(value/(10**12)))+"T"
    else:
        new_val = str(value)
    return new_val


def unlog_plot(values, base):
    to_series = pd.Series(values)
    exponented = base**to_series
    return exponented.map(scale_units).values.tolist()

def full_value_counts(df, column):
    unique = df[column].unique().size
    totalna = df[column].isna().sum()
    percent_na = totalna/df[column].size
    print(f"There are {unique} unique values with {totalna} nan values making up {percent_na*100:.1f}%")
    for value, count in df[column].value_counts().iteritems():
        print(f"{count}-{value}")

# Modifications to masked heatmap parameters from lecture notes.
def trimmed_heatmap(df, columns):
    plt.figure(figsize=(15, 10))
    corr = df[columns].corr()
    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.set_context('talk', font_scale=1)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.95, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    return plt.show()