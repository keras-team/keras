from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from six.moves import range

class PlotGenerator(object):
    def __init__(self,
                 linestyles=['r-', 'b-', 'r:', 'b:'],
                 linestyles_first_epoch=['rs-', 'b^-', 'r:', 'b:'],
                 show_regressions=True,
                 poly_forward_perc=0.1, poly_backward_perc=0.2,
                 poly_n_forward_min=10, poly_n_backward_min=20,
                 poly_degree=1,
                 show_plot_window=True, save_to_filepath=None):
        """Constructs the plotter.
        Args:
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
        self.poly_forward_perc = 0.1
        self.poly_backward_perc = 0.2
        self.poly_backward_min = poly_n_backward_min
        self.poly_forward_min = poly_n_forward_min
        self.poly_degree = poly_degree
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath

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

    def update(self, epoch,
               stats_train_loss, stats_train_acc,
               stats_val_loss, stats_val_acc):
        self._redraw_plot(epoch, stats_train_loss, stats_train_acc, stats_val_loss, stats_val_acc)

        # show plot window or redraw an existing one
        if self.show_plot_window:
            plt.figure(self.fig.number)
            #if epoch == 0:
            plt.show(block=False)
            #else:
            plt.draw()

        # save
        if self.save_to_filepath:
            self._save_plot(self.save_to_filepath)


    def _save_plot(filepath):
        self.fig.savefig(filepath)

    def _redraw_plot(self, epoch,
                    stats_train_loss, stats_train_acc,
                    stats_val_loss, stats_val_acc):
        """Updates the charts in the current plotting window with new values.
        Args:
            epoch: The index of the current epoch, starting at 0.
            stats_train_loss: All of the training loss values of each
                epoch (list of floats).
            stats_train_acc: All of the training accuracy values of each
                epoch (list of floats).
            stats_val_loss: All of the validation loss values of each
                epoch (list of floats).
            stats_val_acc: All of the validation accuracy values of each
                epoch (list of floats).
            save_plot_filepath: The full filepath of the file to which the
                plot is ought to be saved, e.g. "/tmp/plot.png" or None if it
                shouldnt be saved to a file.
        Returns:
            void
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
        handle_tl, = ax1.plot(epochs, stats_train_loss, linestyles[0], label='train loss')
        handle_vl, = ax1.plot(epochs, stats_val_loss, linestyles[1], label='val loss')
        handle_ta, = ax2.plot(epochs, stats_train_acc, linestyles[0], label='train acc')
        handle_va, = ax2.plot(epochs, stats_val_acc, linestyles[1], label='val acc')

        if epoch <= 0 or not self.show_regressions:
            # First epoch, no linear regression yet
            ax1.plot([], [], linestyles[2], label='train loss regression')
            ax1.plot([], [], linestyles[3], label='val loss regression')
            ax2.plot([], [], linestyles[2], label='train acc regression')
            ax2.plot([], [], linestyles[3], label='val acc regression')
        else:
            # second epoch or later => do linear regression
            # for future points of all lines
            
            # Number of epochs in the future.
            # 10% of current epochs (e.g. 20 for 200) and minimum 10.
            n_forward = int(max((epoch+1)*self.poly_forward_perc, self.poly_forward_min))

            # Based on that number of epochs in the past:
            # 20% of current epochs (e.g. 20 for 100) and minimum 20.
            n_backwards = int(max((epoch+1)*self.poly_backward_perc, self.poly_backward_min))

            # Degree of the polynom
            poly_degree = 1

            # List of epochs for which to estimate/predict the likely value.
            # (Not range(e+1, ...) so that the regression line is better
            # connected to the line its based on (no obvious gap).)
            future_epochs = [i for i in range(epoch, epoch + n_forward)]

            # Predicted values for each line
            poly_train_loss = np.poly1d(np.polyfit(epochs[-n_backwards:], stats_train_loss[-n_backwards:], poly_degree))
            poly_val_loss = np.poly1d(np.polyfit(epochs[-n_backwards:], stats_val_loss[-n_backwards:], poly_degree))
            poly_train_acc = np.poly1d(np.polyfit(epochs[-n_backwards:], stats_train_acc[-n_backwards:], poly_degree))
            poly_val_acc = np.poly1d(np.polyfit(epochs[-n_backwards:], stats_val_acc[-n_backwards:], poly_degree))

            # Plot each regression line
            ax1.plot(future_epochs, [poly_train_loss(i) for i in future_epochs], linestyles[2], label='train loss regression')
            ax1.plot(future_epochs, [poly_val_loss(i) for i in future_epochs], linestyles[3], label='val loss regression')
            ax2.plot(future_epochs, [poly_train_acc(i) for i in future_epochs], linestyles[2], label='train acc regression')
            ax2.plot(future_epochs, [poly_val_acc(i) for i in future_epochs], linestyles[3], label='val acc regression')

        # Add legend (below chart)
        ax1.legend(['train loss', 'val loss', 'train loss regression', 'val loss regression'], bbox_to_anchor=(0.7, -0.08), ncol=2)
        ax2.legend(['train acc', 'val acc', 'train acc regression', 'val acc regression'], bbox_to_anchor=(0.7, -0.08), ncol=2)

        # Labels for x and y axis
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('epoch')

        # Show a grid in both charts
        ax1.grid(True)
        ax2.grid(True)
