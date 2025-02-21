# conda create -n diffsbdd python=3.10.4 -y
# conda activate diffsbdd
conda install pytorch=1.12.1 -c pytorch -y
conda install -c conda-forge pytorch-lightning=1.7.4 -y
conda install -c conda-forge wandb=0.13.1 -y
conda install -c conda-forge rdkit=2022.03.2 -y
conda install -c conda-forge biopython=1.79 -y
conda install -c conda-forge imageio=2.21.2 -y
conda install -c anaconda scipy=1.7.3 -y
conda install -c pyg pytorch-scatter=2.1.0 -y
conda install -c conda-forge openbabel=3.1.1 -y
conda install seaborn -y
pip install networkx
conda install torchmetrics=0.11.4 -y
conda install torch-scatter -y
