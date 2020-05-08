!curl https: // course.fast.ai/setup/colab | bash

!wget https: // repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh & & bash Anaconda3-5.2.0-Linux-x86_64.sh - bfp / usr/local
!conda install fire - c conda-forge

!git clone https://github.com/fastai/course-v3.git 
!cp course-v3/nbs/dl2/notebook2script.py /content/
!cp course-v3/nbs/dl2/00_exports.ipynb /content/

!python notebook2script.py 00_exports.ipynb
