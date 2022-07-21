# M1 mac python env setup
- ```
    chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
    sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
    ```
- `conda env create -n rpi_py_env && conda activate rpi_py_env`
- `conda install -c conda-forge tensorflow==2.6.0`
- `pip install opencv-python`