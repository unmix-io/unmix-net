import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import CheckButtons
import os
import glob

results_name = "results.csv"
sources = ["/Volumes/gpu-server/home-muellrap/unmix-net/runs/", "/Volumes/gpu-server/home-kaufman3/unmix-net/runs/"]

result_files = []
for source in sources:
    runs = [os.path.join(source, o) for o in os.listdir(source) if os.path.isdir(os.path.join(source,o))]
    for run in runs:
        result_file = [os.path.join(run, o) for o in os.listdir(run) if os.path.isfile(os.path.join(run,o)) and o == results_name]
        result_files.extend(result_file)

import csv
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool
# output to static HTML file
output_file("lines.html")

# create a new plot with a title and axis labels
p = figure(title="Comparison", sizing_mode='stretch_both')

for i,r in enumerate(result_files):
    with open(r, mode='r') as csv_file:
        run_name = os.path.basename(os.path.dirname(r))
        csv_reader = csv.DictReader(csv_file,  delimiter=';')
        
        loss = []
        val_loss = []
        epochs = []
        for row in csv_reader:
            loss.append(float(row["loss"]))
            val_loss.append(float(row["val_loss"]))
            epochs.append(int(row["epoch"]))

        # add a line renderer with legend and line thickness
        line_loss = p.line(epochs, loss, legend=run_name + "-loss", line_width=2, color=Category20[20][i])
        
        line_val_loss = p.line(epochs, val_loss, legend=run_name + "-val_loss", line_width=2, color=Category20[20][i*2])

p.legend.location = "top_left"
p.legend.click_policy="hide"
# show the results
show(p)





        # ax = plt.axes([0.4, 0.2, 0.4, 0.6] )
        # #fig, ax = plt.subplots()
        

        # l0, = ax.plot(epochs, loss, visible=True, lw=1, label=run_name + '-loss')
        # l1, = ax.plot(epochs, val_loss, visible=True, lw=1, label=run_name+'-val_loss')

    
# ax.legend()
    
# rax = plt.axes([0.1, 0.2, 0.2, 0.6])
# check = CheckButtons(rax, [x[0] for x in plots], [True for x in plots])
# check.on_clicked(show_hide)
# plt.show()




# import pandas as pd

# from bokeh.palettes import Spectral4
# from bokeh.plotting import figure, output_file, show
# from bokeh.sampledata.stocks import AAPL, IBM, MSFT, GOOG

# p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
# p.title.text = 'Click on legend entries to hide the corresponding lines'

# for data, name, color in zip([AAPL, IBM, MSFT, GOOG], ["AAPL", "IBM", "MSFT", "GOOG"], Spectral4):
#     df = pd.DataFrame(data)
#     df['date'] = pd.to_datetime(df['date'])
#     p.line(df['date'], df['close'], line_width=2, color=color, alpha=0.8, legend=name)

# p.legend.location = "top_left"
# p.legend.click_policy="hide"

# output_file("interactive_legend.html", title="interactive_legend.py example")

# show(p)

# t = np.arange(0.0, 2.0, 0.01)
# s0 = np.sin(2*np.pi*t)
# s1 = np.sin(4*np.pi*t)
# s2 = np.sin(6*np.pi*t)

# fig, ax = plt.subplots()
# l0, = ax.plot(t, s0, visible=False, lw=2)
# l1, = ax.plot(t, s1, lw=2)
# l2, = ax.plot(t, s2, lw=2)
# plt.subplots_adjust(left=0.2)

# rax = plt.axes([0.05, 0.4, 0.1, 0.15])
# check = CheckButtons(rax, ('2 Hz', '4 Hz', '6 Hz'), (False, True, True))


# def func(label):
#     if label == '2 Hz':
#         l0.set_visible(not l0.get_visible())
#     elif label == '4 Hz':
#         l1.set_visible(not l1.get_visible())
#     elif label == '6 Hz':
#         l2.set_visible(not l2.get_visible())
#     plt.draw()
# check.on_clicked(func)

# plt.show()