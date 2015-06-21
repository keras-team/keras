from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from six.moves import range


class PlotGenerator(object):
    """Class to plot training and validation loss and accuracy. Usually not
    used directly, but instead together with callbacks.Plotter.

    Possible future improvements for this class:
    - Allow plotting at every nth batch (instead of only at the end of each
        epoch).
    - Allow plotting the results of multiple models in the same chart
        (e.g. to compare multiple trained models with each other).
    - Add Simple Moving Averages.
    - Add Bollinger Bands or something similar.
    - Save a String representation of the used model in the saved image
        (e.g. "Dense 1024 relu, Dense 1024 relu, 10 softmax"). This would
        simplify comparing the results of many different models and estimating
        which one performed best.
    - Save the shape of X and Y in the saved image (same reason as above).
    - Save the following stats in the image (same reason as above):
        Number of epochs run, Number of samples,
        best validation values, time required per batch,
        time required per epoch.
    - Hide the accuracy plot completely if show_accuracy was set to False.
    - Allow to provide save_to_filepath as a lambda function which receives
    the current epoch and batch index."""
    def __init__(self,
                 save_to_filepath=None, show_plot_window=True,
                 linestyles=None,
                 linestyles_first_epoch=None,
                 show_regressions=True,
                 poly_forward_perc=0.1, poly_backward_perc=0.2,
                 poly_n_forward_min=10, poly_n_backward_min=20,
                 poly_degree=1):
        """Constructs the plotter.
        Args:
            save_to_filepath: The filepath to a file at which the plot
                is ought to be saved, e.g. "/tmp/last_plot.png". Set this value
                to None if you don't want to save the plot.
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath.
            linestyles: List of two string values containing the stylings
                of the chart lines. The first value is for the training
                line, the second for the validation line. Loss and accuracy
                charts will both use that styling.
            linestyles_first_epoch: Different stylings for the chart lines
                for the very first epoch (no two points yet to draw a line).
            show_regression: Whether or not to show a regression, indicating
                where each line might end up in the future.
            poly_forward_perc: Percentage value (e.g. 0.1 = 10%) indicating
                for how far in the future each regression line will be
                calculated. The percentage is relative to the current epoch.
                E.g. if epoch is 100 and this value is 0.2, then the regression
                will be calculated for 20 values in the future.
            poly_backward_perc: Similar to poly_forward_perc. Percentage of
                the data basis to use in order to calculate the regression.
                E.g. if epoch is 100 and this value is 0.2, then the last
                20 values will be used to predict the future values.
            poly_n_forward_min: Minimum value for how far in the future
                the regression values will be predicted for each line.
                E.g. 10 means that there will always be at least 10 predicted
                values, even for e.g. epoch 5.
            poly_n_backward_min: Similar to poly_n_forward_min. Minimum
                epochs to use backwards for predicting future values.
            poly_degree: Degree of the polynomial to use when predicting
                future values. Should usually be 1.
        """
        self.linestyles = linestyles
        self.linestyles_first_epoch = linestyles_first_epoch
        self.show_regressions = show_regressions
        self.poly_forward_perc = poly_forward_perc
        self.poly_backward_perc = poly_backward_perc
        self.poly_backward_min = poly_n_backward_min
        self.poly_forward_min = poly_n_forward_min
        self.poly_degree = poly_degree
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath

        if linestyles is None:
            self.linestyles = ['r-', 'b-', 'r:', 'b:']

        if linestyles_first_epoch is None:
            self.linestyles_first_epoch = ['rs-', 'b^-', 'r:', 'b:']

        # ----
        # Initialize plots
        # ----
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))

        # set_position is neccessary here in order to place the legend properly
        box1, box2 = ax1.get_position(), ax2.get_position()
        ax1.set_position([box1.x0, box1.y0 + box1.height * 0.1,
                          box1.width, box1.height * 0.9])
        ax2.set_position([box2.x0, box2.y0 + box2.height * 0.1,
                          box2.width, box2.height * 0.9])

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Updates the plot with the latest data.
        Args:
            train_loss: All of the training loss values of each
                epoch (list of floats).
            train_acc: All of the training accuracy values of each
                epoch (list of floats).
            val_loss: All of the validation loss values of each
                epoch (list of floats).
            val_acc: All of the validation accuracy values of each
                epoch (list of floats).
        """
        self._redraw_plot(epoch, train_loss, train_acc, val_loss, val_acc)

        # show plot window or redraw an existing one
        if self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show(block=False)
            plt.draw()

        # save
        if self.save_to_filepath:
            self._save_plot(self.save_to_filepath)

    def _save_plot(self, filepath):
        """Saves the current plot to a file.
        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath)

    def _redraw_plot(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Redraws the plot with new values.
        Args:
            epoch: The index of the current epoch, starting at 0.
            train_loss: All of the training loss values of each
                epoch (list of floats).
            train_acc: All of the training accuracy values of each
                epoch (list of floats).
            val_loss: All of the validation loss values of each
                epoch (list of floats).
            val_acc: All of the validation accuracy values of each
                epoch (list of floats).
        """

        ax1 = self.ax1
        ax2 = self.ax2

        # List of each epoch (x-axis)
        epochs = list(range(0, epoch+1))

        # Clear loss and accuracy charts
        ax1.clear()
        ax2.clear()

        # Set titles of charts (at the top)
        ax1.set_title('loss')
        ax2.set_title('accuracy')

        # Set the styles of the lines used in the charts
        # r- => red line, b- => blue line,
        # rs- => red line with squares for each data point,
        # b^- => blue line with triangles for each data point
        # Different line style for epochs after the  first one, because
        # the very first epoch has only one data point and therefore no line
        # and would be invisible without the changed style.
        linestyles = self.linestyles if epoch > 0 else self.linestyles_first_epoch

        # Plot the lines
        if train_loss:
            ax1.plot(epochs, train_loss, linestyles[0], label='train loss')
        if val_loss:
            ax1.plot(epochs, val_loss, linestyles[1], label='val loss')
        if train_acc:
            ax2.plot(epochs, train_acc, linestyles[0], label='train acc')
        if val_acc:
            ax2.plot(epochs, val_acc, linestyles[1], label='val acc')

        if self.show_regressions:
            # Number of epochs in the future.
            # 10% of current epochs (e.g. 20 for 200) and minimum 10.
            n_forward = int(max((epoch+1)*self.poly_forward_perc,
                                self.poly_forward_min))

            # Based on that number of epochs in the past:
            # 20% of current epochs (e.g. 20 for 100) and minimum 20.
            n_backwards = int(max((epoch+1)*self.poly_backward_perc,
                                  self.poly_backward_min))

            # List of epochs for which to estimate/predict the likely value.
            # (Not range(e+1, ...) so that the regression line is better
            # connected to the line its based on (no obvious gap).)
            future_epochs = [i for i in range(epoch, epoch + n_forward)]

            self.plot_regression_line(ax1, train_loss, epochs, future_epochs,
                                      n_backwards, linestyles[2],
                                      "train loss regression")
            self.plot_regression_line(ax1, val_loss, epochs, future_epochs,
                                      n_backwards, linestyles[3],
                                      "val loss regression")
            self.plot_regression_line(ax2, train_acc, epochs, future_epochs,
                                      n_backwards, linestyles[2],
                                      "train acc regression")
            self.plot_regression_line(ax2, val_acc, epochs, future_epochs,
                                      n_backwards, linestyles[3],
                                      "val acc regression")

        # Add legend (below chart)
        ax1.legend(['train loss', 'val loss',
                    'train loss regression', 'val loss regression'],
                   bbox_to_anchor=(0.7, -0.08), ncol=2)
        ax2.legend(['train acc', 'val acc',
                    'train acc regression', 'val acc regression'],
                   bbox_to_anchor=(0.7, -0.08), ncol=2)

        # Labels for x and y axis
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')

        # Show a grid in both charts
        ax1.grid(True)
        ax2.grid(True)

    def plot_regression_line(self, plot_ax, data, epochs, future_epochs,
                             n_backwards, linestyle, label):
        """Calculates and plots a regression line.
        Args:
            plot_ax: The ax on which to plot the line.
            data: The data used to perform the regression
                (e.g. training loss values).
            epochs: List of all epochs (0, 1, 2, ...).
            future_epochs: List of the future epochs for which values are
                ought to be predicted.
            n_backwards: How far back to go in time (in epochs) in order
                to compute the regression. (E.g. 10 = calculate it on the
                last 10 values max.)
            linestyle: Linestyle of the regression line.
            label: Label of the regression line.
        """
        # dont try to draw anything if the data list is empty or it's the
        # first epoch
        if len(data) > 1:
            poly = np.poly1d(np.polyfit(epochs[-n_backwards:],
                                        data[-n_backwards:], self.poly_degree))
            future_values = [poly(i) for i in future_epochs]
            plot_ax.plot(future_epochs, future_values, linestyle, label=label)
