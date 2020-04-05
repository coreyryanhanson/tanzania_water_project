import functools
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class Multiplot(object):
    """An object to quickly generate multiple plots for each column in a DataFrame"""

    def __init__(self, df, n_cols=3, figsize=(15, 15)):
        """Sets up the general parameters to be used across all graphs."""

        self.df = df
        self.columns = self.df.columns
        self.figsize = figsize
        self.n_cols = n_cols
        self._set_rows()

    def _multiplot_wrapper(func):
        """Decorator to be used to wrap plotting function to generate and plot
        multiple matplotlib figures and axes objects."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            fig, axes = plt.subplots(nrows=self.n_rows,
                                     ncols=self.n_cols, figsize=self.figsize)
            for i, self.last_col in enumerate(self.columns):
                row, col = i // self.n_cols, i % self.n_cols
                self.last_ax = axes[row][col]
                func(self, *args, **kwargs)
            plt.show()

        return wrapper

    def _set_rows(self):
        """Determines the amount of row axes needed depending on the total
        plots and the column size"""

        self.n_rows = math.ceil(self.columns.size / self.n_cols)

    def column_change(self, columns, drop=True):
        """Allows changes to what columns will be graphed. Default is to drop, but
        can add columns as well."""

        if drop:
            self.columns = self.columns.drop(columns)
        else:
            columns = pd.Index(columns)
            self.columns = self.columns.append(columns)
            self.columns = self.columns.drop_duplicates()
        self._set_rows()

    @_multiplot_wrapper
    def sb_multiplot(self, func, kwargs=None, default_axis=False):
        """Flexible way of calling iterating through plots of a passed
        Seaborn function. Default axis determines what axis the iterated
        variables will take on. Leave blank for one dimensional plots."""

        if default_axis:
            return func(data=self.df, **{default_axis: self.last_col}, ax=self.last_ax, **kwargs)
        else:
            return func(self.df[self.last_col], ax=self.last_ax, **kwargs)

#Changes long numeric values and replaces them with more human readable abbreviations.
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

#Inverts the log functions put on features. To be applied on ticks, so that the scale is visually condensed but the values
# are human readable.
def unlog_plot(values, base):
    to_series = pd.Series(values)
    exponented = base**to_series
    return exponented.map(scale_units).values.tolist()

#Shows the full breadth of possilbe values and nans for a column of a dataframe.
def full_value_counts(df, column):
    unique = df[column].unique().size
    totalna = df[column].isna().sum()
    percent_na = totalna/df[column].size
    print(f"There are {unique} unique values with {totalna} nan values making up {percent_na*100:.1f}%")
    for value, count in df[column].value_counts().iteritems():
        print(f"{count}-{value}")

# Modifications to masked heatmap parameters from lecture notes.
def trimmed_heatmap(df, columns, font_scale=1, annot=True):
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
    sns.set_context('talk', font_scale=font_scale)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.95, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=annot)

    return plt.show()