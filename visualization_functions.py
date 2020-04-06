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
        self.set_cols(n_cols)
        self.linearity_plots = 5

    def _multicol_plot_wrapper(func):
        """Decorator to be used to wrap plotting function to generate and plot
        multiple matplotlib figures and axes objects for multiple columns."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            fig, axes = self._generate_subplots()
            for i, self.last_col in enumerate(self.columns):
                self._determine_ax(axes, i)
                func(self, *args, **kwargs)
            plt.show()

        return wrapper

    def _determine_ax(self, axes, i):
        """Sets current axis based on iterator and axes object. If only one
        column, it does not look for a column index."""

        row, col = i // self.n_cols, i % self.n_cols
        if self.n_cols == 1:
            self.last_ax = axes[row]
        else:
            self.last_ax = axes[row][col]

    def _generate_subplots(self):
        """Creates subplots based on current parameter attributes"""

        return plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=self.figsize)

    def _set_rows(self, n_plots=False):
        """Determines the amount of row axes needed depending on the total
        plots and the column size"""

        if not n_plots:
            n_plots = self.df.columns.size
        self.n_rows = math.ceil(n_plots / self.n_cols)

    def _xyz(self, terms, iterable):
        """Grabs axis values from a dictionary and inserts the iterable into
        the first empty instance. Returns a dictionary of only filled axes."""

        x, y, z = terms.get("x"), terms.get("y"), terms.get("z")
        var_list = [x, y, z]
        for i, var in enumerate(var_list):
            if not var:
                var_list[i] = iterable
                break
        var_dict = {key: value for key, value in zip(["x", "y", "z"], filter(None, var_list))}
        return var_dict

    def _xyz_to_kwargs(self, kwargs, iterable, return_axes=False):
        axes = self._xyz(kwargs, iterable)
        new_kwargs = kwargs.copy()
        new_kwargs.update(axes)
        if return_axes:
            return new_kwargs, axes
        else:
            return new_kwargs

    def modify_col_list(self, columns, drop=True):
        """Allows changes to what columns will be graphed. Default is to drop, but
        can add columns as well."""

        if drop:
            self.columns = self.columns.drop(columns)
        else:
            columns = pd.Index(columns)
            self.columns = self.columns.append(columns)
            self.columns = self.columns.drop_duplicates()
        self._set_rows()

    def set_cols(self, n_cols):
        """Changes the amount of plot columns to display and adjusting the
        rows needed accordingly."""

        self.n_cols = n_cols
        self._set_rows()

    def _plot_qq(self, comparison_df):
        columns = comparison_df.columns
        ax_kwargs = {x: y for x, y in zip(["x", "y"], columns)}
        qq_data = pd.DataFrame(columns=columns)
        for column in columns:
            qq_data[column] = np.quantile(comparison_df[column], np.arange(0, 1, .01))
        return sns.scatterplot(data=qq_data, ax=self.last_ax, **ax_kwargs)

    def _prediction_df(self, predictions, actual):
        columns, pred_list = ["predicted", "actual"], np.stack((predictions, actual))
        return pd.DataFrame(pred_list.T, columns=columns)

    def _sb_linearity_plots(self, comparison_df):
        fig, axes = self._generate_subplots()
        for i in np.arange(self.linearity_plots):
            self._determine_ax(axes, i)
            self._sb_linearity_switch(comparison_df, i)
        plt.show()

    def _sb_linearity_switch(self, comparison_df, i):

        if i == 0:
            self._plot_qq(comparison_df)
        else:
            pass

    def sb_linearity_test(self, column, target):

        self._set_rows(self.linearity_plots)

        reg = LinearRegression()
        y_act = self.df[target]
        reg.fit(self.df[[column]], y_act)
        y_pred = reg.predict(self.df[[column]])
        r_squared = reg.score(self.df[[column]], y_act)
        coef, intercept = reg.coef_[0], reg.intercept_
        comparison_df = self._prediction_df(y_pred, y_act)

        print(y_pred.shape)
        self._sb_linearity_plots(comparison_df)

        # Resets rows to their defaults
        self._set_rows()

    @_multicol_plot_wrapper
    def sb_multiplot(self, func, kwargs=None, default_axis=False):
        """Flexible way of calling iterating through plots of a passed
        Seaborn function. Default axis determines what axis the iterated
        variables will take on. Leave blank for one dimensional plots."""

        if default_axis and kwargs:
            kwargs = self._xyz_to_kwargs(kwargs, self.last_col)
            return func(data=self.df, ax=self.last_ax, **kwargs)
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