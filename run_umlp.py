from ase.optimize import BFGS,FIRE
from ase.filters import ExpCellFilter
#from mattersim.forcefield import MatterSimCalculator
#from sevenn.calculator import SevenNetCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import ase.io
import sys,os

device ="cpu" # Change to "cuda" if you have a GPU and want to use it

# Choose the force field calculator you want to use
#Mattersim
#calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)

#SevenNet
# Set device="cuda" if CUDA/GPU is available
#calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', device=device)

#Orbital
orbff = pretrained.orb_v3_conservative_20_omat(
  device=device,
  precision="float32-high",   # or "float32-highest" / "float64
)
calc = ORBCalculator(orbff, device=device)


def rel(atoms):
    atoms.calc = calc
    ucf = ExpCellFilter(atoms)
    dyn = FIRE(ucf,logfile="fire.log")
    dyn.run(fmax=0.2,steps=200)
    dyn = BFGS(ucf,logfile="bfgs.log")
    dyn.run(fmax=0.04,steps=1000)
    return atoms.get_potential_energy()

if __name__ == '__main__':
    os.chdir(sys.argv[1])
    atoms = ase.io.read('structure.vasp')
    ase.io.write('structure_norel.vasp', atoms)
    energy = rel(atoms)
    with open('energy.dat', 'w') as outfi:
        outfi.write(f'{energy}\n')
    ase.io.write('structure.vasp', atoms)
    
