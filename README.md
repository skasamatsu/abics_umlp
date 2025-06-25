# abICS + 汎用機械学習ポテンシャル　チュートリアル

## 準備

### Python環境の準備
condaやpyenvなどですでに自分で管理できている場合は、本チュートリアル向けに環境を用意してください。Pythonが古いと動作しない可能性があります。3.12で検証済みです。ここでは一例としてminiforgeのcondaをインストールして、環境を用意する方法を記載します。
```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh

conda create -n abics_umlp python=3.12
conda activate abics_umlp
```

### MPIライブラリ
並列計算用のMPIライブラリを用意してください。共同利用計算機の場合はあらかじめ準備されているのが普通です。Ubuntuの場合はapt, Macの場合はhomebrewなどで入れるのが無難でしょう。

### PyTorch
https://pytorch.org/get-started/locally/ を見て、計算機に合っているバージョンをインストールしてください。GPUを使う場合はGPU対応のものを入れてください。Linuxへのインストールの一例：
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### abICS
```sh
pip3 install abics
```

### aenet
aenet 2.0.4をインストールしてください。開発元のバージョンは最近のIntel oneapiだとsegfaultするので、fix版を用意してあります。もう少し古いコンパイラやgfortran, gccを使う場合は開発元のバージョンでも多分大丈夫です。Pythonインタフェースをコンパイルする必要はありません。
```sh
git clone https://github.com/skasamatsu/aenet.git
cd aenet
git checkout fix_optionalrec
```
あとは、READMEのInstallation通りでインストールできるはず。

### 汎用機械学習ポテンシャル
試したいものを入れておいてください。ASEのcalculatorインタフェースで使えるものであれば、すぐに対応可能です。
```sh
pip3 install sevenn
pip3 install orb-models
```

### GNU parallel
訓練データ生成を並列に実行するためのGNU Parallelをインストールしてください。例えば、condaが使える場合、
```sh
conda install parallel
```
でインストール可能です。

### aenet-lammps (optional)
https://issp-center-dev.github.io/abICS/docs/master/ja/html/tutorial/aenet.html の4.2.3に従ってaenet-lammpsをインストールしてください。モンテカルロ計算が数倍加速できます。

### GPUのMPS設定
GPUを用いる場合は、複数のプロセスを同時にGPU上で実行するため、MPS (multi-process service)の設定をしてください。
```sh
export CUDA_VISIBLE_DEVICES=0           # Select GPU 0.
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS    # Set GPU 0 to exclusive mode.
nvidia-cuda-mps-control -d
```
設定をもとに戻すには
```sh
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c 0
```
### チュートリアルファイルのダウンロード
```sh
git clone https://github.com/skasamatsu/abics_umlp.git
cd abics_umlp
```

## 計算の実行(説明省略)
### input.tomlの調整
14行目の
```
path= '/home/kasamatsu/git/aenet/bin/predict.x-2.0.4-ifort_serial'
```
39行目、40行目の
```
generate = '/home/kasamatsu/git/aenet/bin/generate.x-2.0.4-ifort_serial'
train = 'mpirun -n 8 /home/kasamatsu/git/aenet/bin/train.x-2.0.4-ifort_intelmpi'
```
を、自身の環境にインストールしたaenetへのパスで置き換えてください。mpi版のtrainを使う場合は適切（CPUコア数以下）に並列数を設定してください。

### 計算の実行
8コア以上のCPUを有するシステムで以下を実行すると、温度依存性がMC1/DOI.datに出力されるはずです。
```sh
for i in 0 1
do
mpirun -n 8 abics_mlref input.toml >> mlref.out
bash parallel_run.sh 
mpirun -n 8 abics_mlref input.toml >> mlref.out
abics_train input.toml >> train.out
mpirun -n 8 abics_sampling input.toml >> sampling.out
done
```

## 計算の実行（ある程度解説付き）
このチュートリアルでは、用意したインプットファイルを用いて、MgAl<sub>2</sub>O<sub>4</sub>スピネル結晶のMg/Al反転度の温度依存性を計算し、abICSの計算手順を確認します。

### input.tomlの調整
14行目の
```
path= '/home/kasamatsu/git/aenet/bin/predict.x-2.0.4-ifort_serial'
```
39行目、40行目の
```
generate = '/home/kasamatsu/git/aenet/bin/generate.x-2.0.4-ifort_serial'
train = 'mpirun -n 8 /home/kasamatsu/git/aenet/bin/train.x-2.0.4-ifort_intelmpi'
```
を、自身の環境にインストールしたaenetへのパスで置き換えてください。mpi版のtrainを使う場合は適切（CPUコア数以下）に並列数を設定してください。



### 初期データの生成
input.tomlに記載されている結晶格子と組成の情報をもとにランダム配置を生成します。第一原理計算コードを指定すると、それ用の入力ファイルも自動生成できます(VASP, OpenMX, QEに対応)。今回はuser指定のエネルギーソルバーを使うので、座標だけがAL0ディレクトリ以下の子ディレクトリに生成されます。

```sh
mpirun -n 8 abics_mlref input.toml > mlref.out
```

rundirs.txtファイルにディレクトリリストが書き込まれるので、その情報を使って計算を実行します。そして、各ディレクトリにenergy.datファイルを用意し、構造緩和後のエネルギーを書き込みます。今回は、汎用機械学習ポテンシャルを使ってこれを行うコードrun_umlp.pyを用います。GNU Parallelを使って並列実行します。Parallelを実行するスクリプトparallel_run.shの中身は以下のようになっています。

```sh
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
```
GPUを用いる場合はGPUのメモリに載る範囲でなるべく大きなNJOBSを設定してください。CPUを用いる場合はCPUコア数に設定するとよいでしょう。

run_umlp.pyの中身は以下のようになっています。GPUを使う場合はdevice=cudaに設定してください。Macの場合はdevice='mps'にすると速くなるかも？
```python
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
    
```

### 計算結果を共通フォーマットに変換
どのソルバーを使った場合でも、一旦共通フォーマットに変換して、次のステップのon-latticeモデル訓練に供します。
```sh
mpirun -n 8 abics_mlref input.toml >> mlref.out
```

### On-latticeモデルの訓練
以上で計算したデータを用いて、構造緩和前の構造から構造緩和後のエネルギーを予測するaenetモデルを訓練します。
```sh
abics_train input.toml >> train.out
```

### モンテカルロ計算の実行
訓練したon-latticeモデルを使ってモンテカルロ計算を行います。
```sh
mpirun -n 8 abics_sampling >> sampling.out
```
MC0ディレクトリに結果が出力されます。エネルギーや反転度の温度依存性がenergy.dat、DOI.datに出力されます。文献のDOIと比較してみましょう。あまり整合していないはずです。

### モンテカルロ計算から配置を抽出し、再計算
以下のサイクルを繰り返して訓練データを改善していきます。
```sh
mpirun -n 8 abics_mlref input.toml >> mlref.out
bash parallel_run.sh 
mpirun -n 8 abics_mlref input.toml >> mlref.out
abics_train input.toml >> train.out
mpirun -n 8 abics_sampling input.toml >> sampling.out
```
DOIなど、改善されたか確認してみましょう。
