# A-SafeBO
Welcome to the repository for Adaptive Safe Bayesian Optimization (A-SafeBO).

This repository contains code related to the publication: Guk Han, et al. "Adaptive Bayesian Optimization for Fast Exploration Under Safety Constraints." IEEE Access (2023), DOI: 10.1109/ACCESS.2023.3271134.

## Enhanced Features Compared to Raw Repository

In contrast to the raw repository, this version includes **additional implementation details** such as working environment setups and comments for specific scripts, providing a more comprehensive understanding of the code.

## Installation

Ensure you have Python 3.7 installed, and then install the necessary dependencies using pip:
```
pip install pandas==1.3.5
pip install GPy==1.10.0
pip install xlrd==1.2.0
pip install matplotlib==3.5.3
pip install openpyxl==3.1.2
pip install scikit-learn==1.0.2
```

## Usage
The main script for this project is **main.py**, which includes numerous settings. However, for simplicity, a simplified version has been created to demonstrate real-world applications.

To run the simplified version, navigate to the directory A-SafeBO and execute the following command:

```
cd A-SafeBO
python main_simplified.py
```

he results are stored in the folder '/benchmark_map/POWERPLANT/A-SafeBO_{max_iter}N{n_sample_array[0]}'.

For instance, here is a sample instance and its content:

```
Option : Namespace(env_name='POWERPLANT', max_iter=100, num_exp=0, num_smaple=10, start_point=[14.942, 74.8896, 1002.0407, 67.8375], threshold=453.0)
Plot only in 2D environment
init param :[  14.942    74.8896 1002.0407   67.8375], init value ;[[456.08642659]], threshold :[[453.]]
num of samples : 10


total time : 24.011947 s
Cnt FNS : 1, EXP : 89, MAX : 10
Total exploration steps : 100
Safe exploration steps : 96
Unsafe exploration steps : 4
Safe exploration rates : 0.960000
Target best position : [[   5.48   40.07 1019.63   65.62]], result : [[495.76]]
Estimated target best points : [[   4.81948524   62.00946496 1000.52440779   64.69011888]], Estimated target best result : [[489.7378567]]
```