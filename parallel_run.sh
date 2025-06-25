#!/bin/sh

RESTART=OFF # ON or OFF
NJOBS=8
if [ "_$RESTART" = "_ON" ]; then
	RESUME_OPT=--resume-failed
else
	RESUME_OPT=""
fi
export OMP_NUM_THREADS=1
parallel  -j $NJOBS --joblog runtask.log $RESUME_OPT  \
	  -a rundirs.txt "python run_umlp.py {} > output 2>&1"

