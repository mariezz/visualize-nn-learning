# Visualizing the learning process of neural networks

## Context and introduction
Dear students, if you have a background in computer science and are learning about artificial neural networks, this repository is made for you! It contains learning material that aims to increase your understanding of the subject by providing visualizations of a simple network's learning process.

I created this learning material in the context of my Master's thesis at KU Leuven, under supervision of Simiao Lin and Hendrik Blockeel.

## Providing feedback
You can send me your feedback (you can use my email address marie.gh@immie.org), this would be very useful for my master's thesis!

In particular, I would like to know the following:
- What is you background: what do you study, and do/did you have courses on machine learning? If you are a KU Leuven student, did you follow one of "Principle of Machine Learning" or "Machine Learning and Inductive Inference"?
- Do you find this learning material useful?
- Does it have added values compared to the machine learning courses you followed?
- Did you gain new insights?
- Are the notebooks and documents easy to understand? Is it easy to take the code over to make you own experiments?

If you find errors or typos, or if there are things that are unclear, please let me know!

## Contents
This repository contains:
- **notebooks** (files with the .ipynb extension): they guide you through the implementation, provide simple visualizations, and let you experiment by yourself;
- **accompanying documents** (pdf files): they introduce the notebooks, help you undestand the implementation, describe insightful experiments, and explain the results;
- **visualization code** (python file): produces more advanced visualizations (animated versions of the visualizations in the notebooks);
- **videos** that are accelerated recordings of the output of the visualization code;
- a file "basic_neural_network_help_functions.py" that contains functions that are used by the notebooks and the visualization code;
- a file "requirements.txt" that contains a list of the required python packages.

The files whose names start with "1" belong to the first experiment, which introduces a basic neural network and analyzes its learning process. I suggest the following order:
- Read `1A_introduction_basic_neural_network.pdf` together with the first part of the notebook `1_basic_neural_network.ipynb`.
- Read `1B_deeper_look_basic_neural_network.pdf` together with the second part of the first notebook.
- Look at the dynamic visualizations. You can either watch the videos (in particular the parts where the learning process is unstable) or run the visualization code `1_basic_neural_network_dynamic.py` yourself (and optionally experiment with the parameters and other aspects of the code). Read the **documentation** at the beginning of the file, it contains information about the figures and other interesting tips!

[to be continued]

## How to run the code

Running the notebooks and the visualization code on your own computer:
- The file with help functions should be in the same folder as the notebook and the visualization code.
- You need to have `matplotlib`, `jupyterlab`, and `notebook` installed. You can install them in a virtual environment (to avoid installing them system-wide):
    1. Create a python virtual environment in your current folder: `python -m venv venv_basic_nn`
    2. Activate the virtual environment: `source venv_basic_nn/bin/activate` (Linux/macOS)
    3. Install the required packages: `pip install -r requirements.txt`
- You can then run jupyter with `jupyter notebook` and run the notebooks there.
- Alternatively, you can use an IDE like VS Code to run the notebooks.
- You can run the visualization code with `python 1_basic_neural_network_dynamic.py`

Running the notebooks online:
- You can upload the notebooks in Google Colab, then upload the file with help functions (go to the session files in the left bar and upload the file there), and then run the notebook there.

