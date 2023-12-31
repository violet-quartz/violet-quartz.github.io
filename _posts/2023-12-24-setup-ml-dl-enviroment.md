---
layout: post
title:  "Setup ML/DL Development Environment"
date:   2023-12-24 12:08 +0800
categories: tech_blog
---

Here several topics about setting up machine learning/deep learning environemnt are discussed.

## Environment Management
An environment is like an isolated workspace that has its own set of libraries and python versions. This isolation prevents conflicts between different projects requiring different library versions or enen python versions.

### Comparison of differnt tools

|       | conda | venv | virtualenv |
| ----- | ----- | ---- | ----- |
| Flexibility | You can choose python version of your env and are limited to that one. | Python itself is grabbed from the operating system and not installed purposefully on each environment. | **It's allowed to switch between different python versions in one environment.**|
| Compatibility | **Conda can handle complex dependencies and install packages of any language, not just python.** | Libraries and programs that do not belong to the python ecosystem can't be installed. May struggle with complex dependencies. | Like venv, it's python specific and may struggle with complex dependencies | 
| Ease of Use | **Easy to use.** | **Easy to use.** | Has more options that can complecate things. |

Choosing the right environment management tool depends on your needs. For machine learning development, I always choose conda. And to share your work, Docker is another kind of solution.

### Conda
- [Getting started with conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)  
- [Conda cheatsheet](https://conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)
- [Dockhub image with conda pre-installed](https://hub.docker.com/r/continuumio/miniconda3)
- Channels
    - Conda packages are hosted remotely in channels. When installing packages using ```conda install``` or ```conda create```, conda will look for specific packages in channels according to their priority.
    - Example:
        ```
        channels:
        - conda-forge
        - defaults
        channel_priority: strict
        ```
    - The default channels are used if you do not specify any -c parameter during ```conda install``` or ```conda create```. Otherwise, the channels you specify are used. 

        ```
        > conda install -c conda-forge notebook
        ```

    - The ```conda-forge``` channel
        - The conda-forge organization maintains a huge list of packages in the conda-forge channel.
        - There are compatibility problems between conda-forge packages and packages contained in the default conda channels. 
        - An environment should either use conda-forge or not, from creation to destruction. Do not mix and match. If you created it without using the conda-forge channel, then do not add it to the mix halfway. In practice, you can create an environment with conda-forge unless in very specific cases where I found incompatibilities: ```conda create -c conda-forge``` using always conda-forge as the first listed channel.
- A good blog about conda: [Conda: Essential Concepts and Tips](https://towardsdatascience.com/conda-essential-concepts-and-tricks-e478ed53b5b#42cb)

### Jupyter notebook
- Install jupyter notebook and widget
```
conda install -c conda-forge matplotlib
conda install -c conda-forge notebook
# To use %matplotlib widget
conda install -c conda-forge ipyml
!jupyter labextension list  # execute in jupyter notebook to verify.
```
- View documentation of functions: place the cursor on the python function and press shift-Tab