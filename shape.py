# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 17:52:04 2023

@author: Thomas Schincariol
"""

import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Cursor, Button
import numpy as np
from dtaidistance import dtw,ed
import bisect
import math
from scipy.cluster.hierarchy import linkage, fcluster

# =============================================================================
# ShapeFinder for Time series Autoregressive
# =============================================================================

def int_exc(seq_n, win):
    """
    Create intervals and exclude list for the given normalized sequences.

    Args:
        seq_n (list): A list of normalized sequences.
        win (int): The window size for pattern matching.

    Returns:
        tuple: A tuple containing the exclude list, intervals, and the concatenated testing sequence.
    """
    n_test = []  # List to store the concatenated testing sequence
    to = 0  # Variable to keep track of the total length of concatenated sequences
    exclude = []  # List to store the excluded indices
    interv = [0]  # List to store the intervals

    for i in seq_n:
        n_test = np.concatenate([n_test, i])  # Concatenate each normalized sequence to create the testing sequence
        to = to + len(i)  # Calculate the total length of the concatenated sequence
        exclude = exclude + [*range(to - win, to)]  # Add the excluded indices to the list
        interv.append(to)  # Add the interval (end index) for each sequence to the list

    # Return the exclude list, intervals, and the concatenated testing sequence as a tuple
    return exclude, interv, n_test

class Shape():
    """
    A class to set custom shape using a graphical interface, user-provided values or random values.

    Attributes:
        time (list): List of x-coordinates representing time.
        values (list): List of y-coordinates representing values.
        window (int): The window size for the graphical interface.
    """

    def __init__(self, time=len(range(10)), values=[0.5]*10, window=10):
        """
        Args:
            time (int): The initial number of time points.
            values (list): The initial values corresponding to each time point.
            window (int): The window size for the graphical interface.
        """
        self.time = time
        self.values = values
        self.window = window

    def draw_shape(self, window):
        """
        Opens a graphical interface for users to draw a custom shape.

        Args:
            window (int): The window size for the graphical interface.

        Notes:
            The user can draw the shape by clicking on the graph using the mouse.
            The Save button stores the drawn shape data in self.time and self.values.
            The Quit button closes the graphical interface.
        """
        root = tk.Tk()
        root.title("Please draw the wanted Shape")

        # Initialize the plot
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        time_data = list(range(window))
        value_data = [0] * window
        line, = ax.plot(time_data, value_data)
        ax.set_xlim(0, window - 1)
        ax.set_ylim(0, 1)
        ax.set_title("Please draw the wanted Shape")

        def on_button_click(event):
            """
            Callback function for the Save button click event.

            Stores the drawn shape data in self.time and self.values and closes the window.

            Args:
                event: The button click event.
            """
            root.drawn_data = (time_data, value_data)
            root.destroy()

        def on_mouse_click(event):
            """
            Callback function for the mouse click event on the plot.

            Updates the plot when the user clicks on the graph to draw the shape.

            Args:
                event: The mouse click event.
            """
            if event.inaxes == ax:
                index = int(event.xdata + 0.5)
                if 0 <= index < window:
                    time_data[index] = index
                    value_data[index] = event.ydata
                    line.set_data(time_data, value_data)
                    fig.canvas.draw()

        def on_quit_button_click(event):
            """
            Callback function for the Quit button click event.

            Closes the graphical interface.

            Args:
                event: The button click event.
            """
            root.destroy()

        # Add buttons and event listeners
        ax_save_button = plt.axes([0.81, 0.05, 0.1, 0.075])
        button_save = Button(ax_save_button, "Save")
        button_save.on_clicked(on_button_click)

        ax_quit_button = plt.axes([0.7, 0.05, 0.1, 0.075])
        button_quit = Button(ax_quit_button, "Quit")
        button_quit.on_clicked(on_quit_button_click)

        # Connect mouse click event to the callback function
        fig.canvas.mpl_connect('button_press_event', on_mouse_click)
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

        # Create and display the Tkinter canvas with the plot
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add toolbar and protocol for closing the window
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        root.protocol("WM_DELETE_WINDOW", on_quit_button_click)

        # Start the Tkinter main loop
        root.mainloop()

        # Update the shape data with the drawn shape
        value_data=pd.Series(value_data)
        value_data=(value_data - value_data.min()) / (value_data.max() - value_data.min())
        self.time = time_data
        self.values = value_data.tolist()
        self.window = len(time_data)

        # Close the figure to avoid multiple figures being opened
        plt.close(fig)

    def set_shape(self,input_shape):
        try:
            input_shape=pd.Series(input_shape)
            input_shape=(input_shape-input_shape.min())/(input_shape.max()-input_shape.min())
            self.time=list(range(len(input_shape)))
            self.values = input_shape.tolist()
            self.window=len(input_shape.tolist())
        except: 
            print('Wrong format, please provide a compatible input.')
        
    def set_random_shape(self,window):
        value_data=pd.Series(np.random.uniform(0, 1,window))
        value_data=(value_data - value_data.min()) / (value_data.max() - value_data.min())
        self.time=list(range(window))
        self.values = value_data.tolist()
        self.window=len(np.random.uniform(0, 1,window).tolist())

    def plot(self,save=None):
        plt.plot(self.time,self.values,marker='o')
        plt.xlabel('Timestamp')
        plt.ylabel('Values')
        plt.title('Shape wanted')
        plt.ylim(-0.05,1.05)
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()

        
class finder():
    """
    A class to find and predict custom patterns in a given dataset using an interactive shape finder.

    Attributes:
        data (DataFrame): The dataset containing time series data.
        Shape (Shape): An instance of the Shape class used for interactive shape finding.
        sequences (list): List to store the found sequences matching the custom shape.
    """
    def __init__(self,data,Shape=Shape(),sequences=[]):
        """
        Initializes the finder object with the given dataset and Shape instance.

        Args:
            data (DataFrame): The dataset containing time series data.
            Shape (Shape, optional): An instance of the Shape class for shape finding. Defaults to Shape().
            sequences (list, optional): List to store the found sequences matching the custom shape. Defaults to [].
        """
        self.data=data
        self.Shape=Shape
        self.sequences=sequences
        
    def find_patterns(self, metric='euclidean', min_d=0.5, dtw_sel=0, select=True, min_mat=0):
        """
        Finds custom patterns in the given dataset using the interactive shape finder.
    
        Args:
            metric (str, optional): The distance metric to use for shape matching. 'euclidean' or 'dtw'. Defaults to 'euclidean'.
            min_d (float, optional): The minimum distance threshold for a matching sequence. Defaults to 0.5.
            dtw_sel (int, optional): The window size variation for dynamic time warping (Only for 'dtw' mode). Defaults to 0.
            select (bool, optional): Whether to include overlapping patterns. Defaults to True.
            min_mat (int, optional): The minimum number of matching sequences. Defaults to 0.
        """
        # Clear any previously stored sequences
        self.sequences = []
        
        # Check if dtw_sel is zero when metric is 'euclidean'
        if metric=='euclidean':
            dtw_sel=0
    
        # Extract individual columns (time series) from the data
        seq = []
        for i in range(len(self.data.columns)): 
            seq.append(self.data.iloc[:, i])
    
        # Normalize each column (time series)
        seq_n = []
        for i in seq:
            seq_n.append((i - i.mean()) / i.std())
    
        # Get exclude list, intervals, and a testing sequence for pattern matching
        exclude, interv, n_test = int_exc(seq, self.Shape.window)
    
        # Convert custom shape values to a pandas Series and normalize it
        seq1 = pd.Series(data=self.Shape.values)
        if seq1.var() != 0.0:
            seq1 = (seq1 - seq1.min()) / (seq1.max() - seq1.min())
        else :    
            seq1 = [0.5]*len(seq1)
        seq1 = np.array(seq1)
    
        # Initialize the list to store the found sequences that match the custom shape
        tot = []
    
        if dtw_sel == 0:
            # Loop through the testing sequence
            for i in range(len(n_test)):
                # Check if the current index is not in the exclude list
                if i not in exclude:
                    seq2 = n_test[i:i + self.Shape.window]
                    if seq2.var() != 0.0:
                        seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                    else:
                        seq2 = np.array([0.5]*len(seq2))
                    try:
                        if metric == 'euclidean':
                            # Calculate the Euclidean distance between the custom shape and the current window
                            dist = ed.distance(seq1, seq2)
                        elif metric == 'dtw':
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2,use_c=True)
                        tot.append([i, dist, self.Shape.window])
                    except:
                        # Ignore any exceptions (e.g., divide by zero)
                        pass
        else:
            # Loop through the range of window size variations (dtw_sel)
            for lop in range(int(-dtw_sel), int(dtw_sel) + 1):
                # Get exclude list, intervals, and a testing sequence for pattern matching with the current window size
                exclude, interv, n_test = int_exc(seq_n, self.Shape.window + lop)
                for i in range(len(n_test)):
                    # Check if the current index is not in the exclude list
                    if i not in exclude:
                        seq2 = n_test[i:i + int(self.Shape.window + lop)]
                        if seq2.var() != 0.0:
                            seq2 = (seq2 - seq2.min()) / (seq2.max() - seq2.min())
                        else:
                            seq2 = np.array([0.5]*len(seq2))
                        try:
                            # Calculate the Dynamic Time Warping distance between the custom shape and the current window
                            dist = dtw.distance(seq1, seq2)
                            tot.append([i, dist, self.Shape.window + lop])
                        except:
                            # Ignore any exceptions (e.g., divide by zero)
                            pass
    
        # Create a DataFrame from the list of sequences and distances, sort it by distance, and filter based on min_d
        tot = pd.DataFrame(tot)
        tot = tot.sort_values([1])
        tot_cut = tot[tot[1] < min_d]
        toti = tot_cut[0]
    
        if select:
            n = len(toti)
            diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
            diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
            diff_df = abs(diff_df)
            tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
    
        if len(tot_cut) > min_mat:
            # If there are selected patterns, store them along with their distances in the 'sequences' list
            for c_lo in range(len(tot_cut)):
                i = tot_cut.iloc[c_lo, 0]
                win_l = int(tot_cut.iloc[c_lo, 2])
                exclude, interv, n_test = int_exc(seq_n, win_l)
                col = seq[bisect.bisect_right(interv, i) - 1].name
                index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                obs = self.data.loc[index_obs:, col].iloc[:win_l]
                self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
        else:
            flag_end=False
            min_mat_loop=min_mat
            while flag_end==False:
                tot_cut = tot.iloc[:min_mat_loop,:]
                toti = tot_cut[0]
                if select:
                    n = len(toti)
                    diff_data = {f'diff{i}': toti.diff(i) for i in range(1, n + 1)}
                    diff_df = pd.DataFrame(diff_data).fillna(self.Shape.window)
                    diff_df = abs(diff_df)
                    tot_cut = tot_cut[diff_df.min(axis=1) >= (self.Shape.window / 2)]
                if len(tot_cut) > min_mat:
                    for c_lo in range(len(tot_cut)):
                        i = tot_cut.iloc[c_lo, 0]
                        win_l = int(tot_cut.iloc[c_lo, 2])
                        exclude, interv, n_test = int_exc(seq_n, win_l)
                        col = seq[bisect.bisect_right(interv, i) - 1].name
                        index_obs = seq[bisect.bisect_right(interv, i) - 1].index[i - interv[bisect.bisect_right(interv, i) - 1]]
                        obs = self.data.loc[index_obs:, col].iloc[:win_l]
                        self.sequences.append([obs, tot_cut.iloc[c_lo, 1]])
                    flag_end=True
                else:
                    min_mat_loop=min_mat_loop+1

        
    def plot_sequences(self,how='units',save=None):
        """
        Plots the found sequences matching the custom shape.

        Args:
            how (str, optional): 'units' to plot each sequence separately or 'total' to plot all sequences together. Defaults to 'units'.

        Raises:
            Exception: If no patterns were found, raises an exception indicating no patterns to plot.
        """
        # Check if any sequences were found, otherwise raise an exception
        if len(self.sequences) == 0:
            raise Exception("Sorry, no patterns to plot.")
    
        if how == 'units':
            # Plot each sequence separately
            for i in range(len(self.sequences)):
                plt.plot(self.sequences[i][0], marker='o')
                plt.xlabel('Date')
                plt.ylabel('Values')  # Corrected typo in xlabel -> ylabel
                plt.suptitle(str(self.sequences[i][0].name), y=1.02, fontsize=15)
                plt.title("d = " + str(self.sequences[i][1]), style='italic', color='grey')
                plt.show()
    
        elif how == 'total':
            # Plot all sequences together in a grid layout
            num_plots = len(self.sequences)
            grid_size = math.isqrt(num_plots)  # integer square root
            if grid_size * grid_size < num_plots:  # If not a perfect square
                grid_size += 1
    
            subplot_width = 7
            subplot_height = 5
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(subplot_width * grid_size, subplot_height * grid_size))
    
            if num_plots > 1:
                axs = axs.ravel()
            if not isinstance(axs, np.ndarray):
                axs = np.array([axs])
    
            for i in range(num_plots):
                axs[i].plot(self.sequences[i][0], marker='o')
                axs[i].set_xlabel('Date')
                axs[i].set_title(f"{self.sequences[i][0].name}\nd = {self.sequences[i][1]}", style='italic', color='grey')
    
            if grid_size * grid_size > num_plots:
                # If there are extra subplot spaces in the grid, remove them
                for j in range(i + 1, grid_size * grid_size):
                    fig.delaxes(axs[j])
    
            plt.tight_layout()
            if save is not None:
                plt.savefig(save, dpi=300, bbox_inches='tight')
            plt.show()

    def create_sce(self,horizon=6,clu_thres=3):
        """
        Creates scenarios based on matched series in historical data.
        
        Args:
            horizon (int): The number of future time steps to consider for scenario creation.
            clu_thres (int): The threshold for clustering, influencing the number of clusters.
        
        """
        # Ensure sequences exist before proceeding
        if len(self.sequences) == 0:
            raise Exception('No shape found, please fit before predict.')
    
        # Extract key stats from stored sequences
        tot_seq = [
            [series.name, series.index[-1], series.min(), series.max(), series.sum()] 
            for series, weight in self.sequences]
    
        pred_seq = []
        # Generate future sequences for each stored sequence
        for col, last_date, mi, ma, somme in tot_seq:
            date = self.data.index.get_loc(last_date)  # Get index position of the last known date
            # Ensure there are enough future values for the specified horizon
            if date + horizon < len(self.data):
                # Extract future values for the given column
                seq = self.data.iloc[date + 1 : date + 1 + horizon, self.data.columns.get_loc(col)].reset_index(drop=True)
                # Normalize sequence using min-max scaling
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
    
        # Convert sequences to a DataFrame
        tot_seq = pd.DataFrame(pred_seq)
        # Perform hierarchical clustering
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon / clu_thres, criterion='distance')
        # Assign clusters to the sequences
        tot_seq['Cluster'] = clusters
        # Compute mean values per cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        # Set the index to the relative frequency of each cluster
        val_sce.index = round(pd.Series(clusters).value_counts(normalize=True).sort_index(), 2)
        # Store the computed scenarios
        self.val_sce = val_sce
        
    def plot_scenario(self,save=None):
        """
        Plots the scenarios associated with their probability.

        """
        # ensure there is data to plot
        if len(self.val_sce) == 0:
            raise Exception('No scenarios found, please create them before plotting.')
        # Calculate vertical (y-axis) and horizontal (x-axis) bounds
        h_max = max(1,self.val_sce.max().max())
        h_min = max(0,self.val_sce.min().min())
        w_min=0
        w_max= len(self.Shape.values)+len(self.val_sce.columns)-1
        # Determine figure size dynamically based on the data span
        width_span = w_max - w_min
        height_span = h_max - h_min
        fig_width = width_span
        fig_height = height_span * 3

        plt.figure(figsize=(fig_width, fig_height))
        # Plot each scenario
        for sce in range(len(self.val_sce)):
            seq = pd.Series([self.Shape.values[-1]]+self.val_sce.iloc[sce,:].tolist())
            seq.index = range(len(self.Shape.values)-1,len(self.Shape.values)+len(seq)-1)
            plt.plot(seq,label=f'P={self.val_sce.index[sce]}',marker='o')
        # Plot input shape
        plt.plot(pd.Series(self.Shape.values),color='grey',marker='o')
        plt.legend()
        plt.title("Scenarios")
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict(self, horizon=6, clu_thres=3):
        """
        Predicts future values based on historical sequences using hierarchical clustering.
    
        Args:
            horizon (int): The number of future time steps to predict.
            clu_thres (int): The threshold for clustering, affecting the number of clusters.
    
        Returns:
            pd.Series: The final predicted sequence.
        """
        # Ensure sequences exist before proceeding
        if len(self.sequences) == 0:
            raise Exception('No shape found, please fit before predict.')
        # Extract key statistics from stored sequences
        tot_seq = [
            [series.name, series.index[-1], series.min(), series.max(), series.sum()] 
            for series, weight in self.sequences]
    
        pred_seq = []
        # Generate future sequences for each stored sequence
        for col, last_date, mi, ma, somme in tot_seq:
            date = self.data.index.get_loc(last_date)  # Get index position of the last known date
            if date + horizon < len(self.data):
                # Extract future values for the given column
                seq = self.data.iloc[date + 1 : date + 1 + horizon, self.data.columns.get_loc(col)].reset_index(drop=True)
                # Normalize sequence using min-max scaling
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())

        tot_seq = pd.DataFrame(pred_seq)
        # Perform hierarchical clustering
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon / clu_thres, criterion='distance')
        # Assign clusters to the sequences
        tot_seq['Cluster'] = clusters
        # Compute mean values per cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        # Determine the most frequent cluster
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(), 2)
        # Extract the cluster with the highest frequency
        pred_ori = val_sce.loc[pr == pr.max(), :]
        # Compute the mean prediction across sequences in the most frequent cluster
        pred_ori = pred_ori.mean(axis=0)
        # Retrieve original shape values for denormalization
        seq1 = pd.Series(data=self.Shape.values)
        # Denormalize predictions back to original scale
        preds = pred_ori * (seq1.max() - seq1.min()) + seq1.min()
    
        return preds





