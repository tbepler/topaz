**<details><summary>Click here to install *from source*</summary><p>**

_Recommended: install Topaz into a virtual Python environment_  
See https://conda.io/docs/user-guide/tasks/manage-environments.html or https://virtualenv.pypa.io/en/stable/ for setting one up.

\
**Install the dependencies**

Tested with python 3.6 and 2.7

- pytorch (>= 1.0.0)
- torchvision
- pillow (>= 6.2.0)
- numpy (>= 1.11)
- pandas (>= 0.20.3) 
- scipy (>= 0.19.1)
- scikit-learn (>= 0.19.0)

Easy installation of dependencies with conda
```
conda install numpy pandas scikit-learn
conda install -c pytorch pytorch torchvision
```
For more info on installing pytorch for your CUDA version see https://pytorch.org/get-started/locally/

\
**Download the source code**
```
git clone https://github.com/tbepler/topaz
```

\
**Install Topaz**

Move to the source code directory
```
cd topaz
```

By default, this will be the most recent version of the topaz source code. To install a specific older version, checkout that commit. For example, for v0.1.0 of Topaz:
```
git checkout v0.1.0
```
Note that older Topaz versions may have different dependencies. Refer to the README for the specific Topaz version.

Install Topaz into your Python path including the topaz command line interface
```
pip install .
```

To install for development use
```
pip install -e .
```

</p></details>
<p>