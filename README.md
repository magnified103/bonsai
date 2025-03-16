# Bonsai Network Generator

Requirements:
- python 3.9

## Install and use as cmd line program


1. ``pip install torch~=2.2.2 numpy~=1.26.3 --index-url https://download.pytorch.org/whl/cpu``
2. ``pip install torch-cluster~=1.6.3 -f https://data.pyg.org/whl/torch-2.2.2+cpu.html``
3. ``pip install -r requirements.txt`` to install dependencies
4. run as ``python -m main --net_config <network-config>.yaml --output_dir <output_dir>``
5. To leave the virtual environment: ``deactivate``


## Use as a python library

Install on your python project this dependency:
``` bonsai @ git+https://codelab.fct.unl.pt/di/computer-systems/bonsai.git@main```


