## Optimal control with POD-Galerkin and sPOD-Galerkin methods

This repo contains the source code for investigating the optimal control problems with reduced order models like the POD_Galerkin and the sPOD-Galerkin methods. There are 5 script files present in the source code that could be run to reproduce the results and play around with the code. These script files are: 
* `FOM.py` solves the optimal control problem by considering the Full Order Model (FOM).
* `PODG_FOTR.py` solves the optimal control problem by considering the POD-Galerkin method under the FOTR framework.
* `PODG_FRTO.py` solves the optimal control problem by considering the POD-Galerkin method under the FRTO framework.
* `sPODG_FOTR.py` solves the optimal control problem by considering the sPOD-Galerkin method under the FOTR framework.
* `sPODG_FRTO.py` solves the optimal control problem by considering the sPOD-Galerkin method under the FRTO framework.

Each of these script files is commented and streamlined for understandability. The reader can play around with the parameters listed in `kwargs` dictionary mentioned in each of the script files. 
