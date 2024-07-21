# Hidden Layer Visualization while Training
A tool to visualize the hidden layers of the model in real-time. 
In this project, we delve in a fundamental yet often overlooked aspect - visualizing the layers of the neural network.

By visually inspecting these weights during training or inference, we can gain insights into overfitting, underfitting, convergence issues, and even help in designing better architectures. 
Visualizing the weights of neural network layers can reveal patterns, identify bottlenecks, and guide us towards more efficient models.


## Visualization of all the weights
![Weight Visualization](/images/weights.gif)

## Visualization of all the layers and their activations
![Layer Visualization](/images/layers.gif)

## How to use this repo
1. Clone the repository:
   ```bash
   git clone https://github.com/strcoder4007/Training-Visualization.git
   cd Training-Visualization
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This command will install all necessary packages including `Pytorch`, `opencv-python`, `imagio`, and `pickle`.

3. Dataset is present in the root of this repo, to run the model, run `main.py`:
   ```bash
   python main.py
   ```
   The `.pkl` data file with all the weights and biases of the model for every epoch and step will be generated and saved in the root of the folder

4. `main.py` will use the `.pkl` file to load all the weights and biases to display them using `Tkinter`