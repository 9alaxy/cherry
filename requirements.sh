#Cherry Requirements
pip install cudatoolkit=11.8.0
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter=2.1.2+pt20cu118
pip install torch-sparse=0.6.18+pt20cu118
pip install torch-summary=1.4.5
# https://www.dgl.ai/pages/start.html
conda install dgl=2.0.0+cu118
pip install tqdm=4.65.0
pip install ogb=1.3.6
pip install matplotlib
pip install seaborn=0.13.2
pip install tabulate=0.9.0
pip install pymetis=2023.1.1