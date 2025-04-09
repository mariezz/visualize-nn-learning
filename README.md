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
- notebooks (files with the .ipynb extension): they guide you through the implementation, provide simple visualizations, and let you experiment by yourself;
- accompanying documents (pdf files): they introduce the notebooks, help you undestand the implementation, describe insightful experiments, and explain the results;
- visualization code (python file): produces more advanced visualizations (animated versions of the visualizations in the notebooks);
- a file "basic_neural_network_help_functions.py" that contains functions that are used by the notebooks and the visualization code.

The files whose names start with "1" belong to the first experiment, which introduces a basic neural network and analyzes its learning process. I suggest the following order:
- Read `1A_introduction_basic_neural_network.pdf` together with the first part of the notebook `1_basic_neural_network.ipynb`.
- Read `1B_deeper_look_basic_neural_network.pdf` together with the second part of the first notebook.
- Run the visualization code `1_basic_neural_network_dynamic.py`. Read the **documentation** at the beginning of the file!

[to be continued]

## How to run the code

Run the notebooks:
- You can use jupyter or an IDE like VS Code to run the notebooks on your computer. You will need to have `matplotlib`,  `jupyterlab`, and `notebook` installed. The file with help functions must be in the same folder as the notebook.
- Alternatively, you can upload the notebook in google colab, then upload the file with help functions (go to the session files in the left bar and upload the file there), and then run the notebook.

Run the visualization code:
- On your computer: you only need `matplotlib` installed and the file with help functions in the same folder.

