---
layout: post
title:  "Python Practice"
date:   2023-12-07 21:55 +0800
categories: practice
type: note
subtype: python
excerpt_separator: $$end_excerpt$$
---
- Create virual environment for isolation
```console
> py -m venv your_env_name
> venv your_env_name\Scripts\activate
> deactivate
```
$$end_excerpt$$
- Install packages
```console
> pip list # check packages installed
> pip install package-name
> pip list
> pip freeze > requirements.txt # generate package requirements automatically
> pip install -r requirements.txt
> pip uninstall -r requirements.txt # remove one by one
> pip uninstall -r requirements.txt -y # remove all at once
> pip install new-package
> pip freeze > requirements.txt # update requirements
``` 
