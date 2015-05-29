#The Nerv Toolkit User Manual#
This user manual will information about how to use __Nerv__ and __Nerv__'s interface.

##How to use##
First make sure you have __lua__ and __CUDA__ installed on your computer.  
__Nerv__ is currently developed via github.You can download and make __Nerv__ by doing the following:
```
cd ~
git clone https://github.com/Determinant/nerv.git
cd nerv
git submodule init && git submodule update
make
```
The `git submodule` command is for the __luajit__ repository inside __Nerv__.  
Now, you can try to run some example scripts.  
```
./nerv examples/cumatrix_example.lua
```

##How to contribute##
Fork the original repository, then use the __pull&merge__ function in github to contribute.  
The pull&merge request can be found on your dashboard in github. See this [sync-help] to sync with the original repository.

##Nerv Packages##
* __luaT__  
Nerv uses [luaT]\(a [Torch] library\) to define lua class in C.
* __[The Nerv utility functions](doc/nerv.md)__  
Inlcudes some utility functions from luaT to implement __Nerv.Class__.
* __The Nerv Matrix Package__  
The Matrix package is a basic package in __Nerv__ that is used to store and manipulate matrices.

[luaT]:https://github.com/torch/torch7/tree/master/lib/luaT
[Torch]:https://github.com/torch
[sync-help]:https://help.github.com/articles/syncing-a-fork/