# Github repo template for scientific projects


## Python environment

Use mamba to manage environments, it allows installing binary packages and is faster than conda.
Save mamba and environemnts in `/data.nst/username/envs/`, as this folder won't be backed up.
(Backup of environments is very time-intensive because of all the small files)

When using the global conda installation from `/usr/ds/` you can set the path for environments in your `/home/username/.condarc` by adding
```console
envs_dirs:
  - /data.nst/username/envs
```


## Create a repository

Simply download the template and copy it to your new repository. Get to know all the files.
There are sompe placeholder in .gitignore and pyproject.toml, which you should replace with your package_name

General approach is that you have a source folder, which we gave directly a name `example_package`, 
and a folder for the notebooks, which we called `notebooks`. In the source folder, 
put code that is reusable, has a clear purpose and some docstrings such that other people
can understand the API. In the notebooks folder, put the plotting code and other code
to perform you digital experiments/analysis. You can also create a `scripts` folder if 
you don't like to work with notebooks.

We can recommend this tutorial regarding scientific python programming: 
[youtu.be/x3swaMSCcYk?si=nhJCxy8UudLPk67k](https://youtu.be/x3swaMSCcYk?si=nhJCxy8UudLPk67k) 
for a general introduction to the topic. It is from a former member of our institute. 


### Installation of the pacakge

The package has to be installed in order to import it in the notebooks or scipts. For 
this the pyproject.toml is required. See [ianhopkinson.org.uk/2022/02/understanding-setup-py-setup-cfg-and-pyproject-toml-in-python/](https://ianhopkinson.org.uk/2022/02/understanding-setup-py-setup-cfg-and-pyproject-toml-in-python/)
for some background information.
When you are inside the repository folder, you can install the package with the following commands:

```console
mamba create -n example_package_env python
mamba activate example_package_env
pip install -e .
```

The -e flag installs the package in editable mode, so you can change the code 
and the changes will be reflected in the package.
It installs all packages specified in pyproject.toml -> project -> dependencies.

and you might also want to install a jupyter environment, for instance

```console
pip install jupyterlab
```

Jupyterlab is a more modern version of the jupyter notebook.
You can start it with:

```console
jupyter-lab
```

Or if you start it on a remote server, you have to allow connections to the server, e.g., by

```console
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser
```

The port can be chosen freely, but it has not to be occupied by another service.

For the installation of optional dev packages specified in 
pyproject.toml -> project.optional-dependencies, you can use

```console
pip install -e .[dev]
```

## Publication

If you publish a paper, you should also publish the code that is necessary to reproduce the results.
This involves saving the versions of the packages you used. For instance, save the
the output of `mamba env export` in a file `environment.yml` in the repository, and/or
`pip freeze > requirements.txt` to save the versions of the packages you used.


