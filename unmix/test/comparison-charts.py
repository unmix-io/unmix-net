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
from bokeh.models import HoverTool, TextInput
from bokeh.models.callbacks import CustomJS

# output to static HTML file
output_file("lines.html")

# create a new plot with a title and axis labels
p = figure(title="Comparison", sizing_mode='stretch_both')
lines = []

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
        line_loss = p.line(epochs, loss, legend=run_name + "-loss", line_width=2, color=Category20[20][i % 20], visible=False)
        line_val_loss = p.line(epochs, val_loss, legend=run_name + "-val_loss", line_width=2, color=Category20[20][i*2 % 20], visible=False)
        lines.append(line_loss)
        lines.append(line_val_loss)


p.legend.location = "top_left"
p.legend.click_policy="hide"
# show the results

text = TextInput(width=300, height=40, height_policy="fixed")

callback = CustomJS(args=dict(p=p, lines=lines, legend=p.legend, text=text), code="""
// the model that triggered the callback is cb_obj:
var b = cb_obj.value;
console.log(cb_obj)
console.log(p);
window.legend = legend;
window.legend[0].items.forEach((l) => {
    if(!l._label)
        l._label = l.label;
    l.label = new RegExp(text.value).test(l._label.value) ? l._label : "";
});
""")
text.js_on_change('value', callback)

from bokeh.layouts import column, row
layout = row(text, p)

show(layout)