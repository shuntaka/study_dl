#!/bin/bash

for i in `seq 0 12`
do
  echo "[$i]" ` date '+%y/%m/%d %H:%M:%S'` "connected."
  open https://colab.research.google.com/github/fastai/course-v3/blob/master/nbs/dl1/lesson3-camvid.ipynb#scrollTo=NuN91awgklFs
  sleep 3600
done