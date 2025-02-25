# Bonsai Network Generator

Requirements:
    - python 3.9

## Install and use as cmd line program


1. ``git clone <repo>``
2. ``cd <repo>``
3. ``pip install virtualenv``(if not already installed)
4. ``virtualenv venv``to create a new environment (called 'venv')
5. ``source venv/bin/activate`` to enter the virtual environment
6. ``pip install -r requirements.txt`` to install dependencies
7. run as ``python -m main --net_config <network-config>.yaml --output_dir <output_dir>``
8. To leave the virtual environment: ``deactivate``


## Use as a python library

Install on your python project this dependency:
``` bonsai @ git+https://codelab.fct.unl.pt/di/computer-systems/bonsai.git@main```


