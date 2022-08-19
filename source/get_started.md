# 快速入门

下面的例子将帮助你快速进行HiNPS的安装和运行，我们还将通过一个例子向你展示HiNPS求解偏微分方程的过程。
## 安装

首先从GitHub获取我们的源代码
```bash
git clone https://github.com/chaoyanggroup/hinps
cd hinps
```

我们推荐使用`conda`进行环境配置。创建一个名为`hinps`的python3.8环境。
```bash
conda activate
conda create -n hinps python==3.8
```

请根据你的安装环境首先安装[PyTorch](https://pytorch.org/)(版本要求>=1.10)。需要使用GPU则请将cudatoolkit=11.3改为你的CUDA版本号。
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
如果只需要在CPU上运行，请使用
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

接下来准备MPI。你需要首先在你的机器上安装MPI，集群上可以通过`module avail`检查是否存在已安装好的MPI，如果有的话请加载。

如果没有我们推荐使用OpenMPI。
```bash
conda install openmpi mpi4py
```

随后安装mpi4py
```bash
conda install mpi4py
```
### 安装HiNPS

在HiNPS文件夹下，可以通过setup.py进行安装。

```bash
python setup.py install
```

如果上述过程出错了，可以手动安装以下依赖包

```bash
pip install sympy matplotlib pandas scipy
```

进入python，检查HiNPS版本从而检查安装是否正确
```
python
import hinps
hinps.__version__  #输出应为当前版本号(0.1)
```

## 第一个算例

下面我们通过一个算例解释如何使用HiNPS进行偏微分方程的求解。我们选择的是一个定义在$T\times \Omega$上的热传导问题，其中$T =[0,1],\Omega=[0,1]^3$。给定狄利克雷边界条件和初始温度条件。我们需要编写运行脚本的代码来表示下述的问题。

$$
\frac{\partial u}{\partial t}- \Delta u = r,~~~~ on ~~\Omega
$$

$$
u = r_d,~~~~ on ~~\partial\Omega
$$

$$
u_{|t=0} = r_i,~~~~ on ~~\partial\Omega
$$


首先我们进行初始化，在这里我们会解析来自[命令行的参数](bash脚本)，返回的args则包括了这些参数。

```python
import hinps as np
...
args = hp.init()
```

### 定义求解区域

准备好求解区域的定义。我们把求解区域的定义分成两部分，一部分是空间区域的定义。在这个例子中我们选择的是一个边长平行于坐标轴的方块形空间区域，所以只需要定义上界`sup`和下界`inf`即可。

```python
geo = hp.geometry.Box(sup=np.array([1., 1., 1.]), inf=np.array([0., 0., 0.]), dim=3)
```
> TIPS: 更多形状、参数详细定义请参考[hinps.geometry](api.md#geometry)

接下来定义时间维度。如果你的问题是诸如对稳态的求解，与时间无关，那么可以跳过这一步。我们将时间维度的信息存在`Dataset`数据结构中，所以你首先需要用新建的几何区域`geo`来构造一个`Dataset`。然后再用`add_time.range`声明你需要加上一个时间维度，传入你关心的时间上界和下界。

```python
train_dataset = hp.data.DecomposedTrainDataset(geometry=geo)
train_dataset.add_time_range(time_inf=0, time_sup=1)
train_dataset.decompose_init(time_dims=1)
```
> 当前我们默认所有的训练任务都需要区域分解，所以我们需要使用`DecomposedTrainDataset`，并且在声明时间维度后需要现式地初始化区域分解信息`decompose_init()`

### 定义问题

接下来我们定义问题。问题分为两部分，方程本身与边界条件。
方程本身可以使用我们内置的`pde`模块。当然我们也支持自定义PDE，请参考[hinps.pde](api.md#pde)
```python
heat = hp.pde.Heat(a=1, b=None, dim=3, device=args.device)
```
另外，我们使用[SymPy](https://www.sympy.org/en/index.html)作为我们符号运算的工具，经过`hinps.expression.Function`的封装，可以变为可调用的函数。在这里你可以根据你习惯的数学表达式写法描述你的函数。接下来得到的`ground`则是一个输入为4维，输出为1维的函数。

```python
from sympy import *
t, x1, x2, x3 = symbols('t x1 x2 x3')
u = (x1**3 - 3 * x1) * (x2**3 - 3 * x2) * (x3**3 - 3 * x3) * exp(-1 * t * (x1 + x2 + x3))
ground = hp.expression.Function(input=[t, x1, x2, x3], output=[u])
# y = ground(x), x.shape == [...,4], y.shape==[..., 1]
```


在HiNPS中，我们认为PDE是定义在求解区域内部的一种“约束”，所以我们构造一个`Constraint`的对象，并且将其加入到`train_dataset`中以表示这个约束。它表示作用于集合空间的内部（以区别于“边界”），类型是"internal"。

"internal"类型的约束是等式约束的形式，左端项(lhs, left-hand side)与我们神经网络推理的结果有关，我们给定一个heat.lhs；右端项(rhs, right-hand side)是我们已知的函数（例如热源、振动源等等）有关。

我们还需要指定这个约束的权重（默认为1），针对这个约束，我们要采样的数目`sample_size`。为了方便调试，我们可以给这个约束一个名字，例如'Internal'。
```python
train_dataset.add_constraint(
    hp.data.Constraint(
        type='internal',
        on_boundary=False,
        lhs_handle=heat.lhs,
        rhs_handle=rhs, # hp.expression.Function
        weight=1.,
    ),
    sample_size=2048,
    name='Internal')
```

我们再定义边界条件。类型为"dirichlet"，也是一种等式约束，但是等式的左边为待求的函数，右边才是我们已知的函数。

我们可以用filter参数来指定该约束作用于边界的哪些部分，如果不指定，则默认该约束作用域整个边界。

```python
train_dataset.add_constraint(
    hp.data.Constraint(
        on_boundary=True,
        type='dirichlet',
        filter=None,  #hp.expression.Function(input=[x1, x2, x3], output=[Abs(x1 - inf[1]) < 1e-8]),
        rhs_handle=r_b, # hp.expression.Function
        weight=1.),
    sample_size=256,
    name='Boundary',
```


接下来我们定义初始条件。关于时间的约束我们需要指定作用的时间范围，对于初始条件，我们让这个范围的上界和下界都等于TIME_INF（初始时刻）就可以了。该参数功能类似边界条件的`filter`。

```python
train_dataset.add_constraint(
    hp.data.Constraint(
        type='dirichlet',
        time_span=(TIME_INF, TIME_INF),
        on_boundary=False,
        rhs_handle=r_i, # hp.expression.Function
        weight=1.,
    ),
    sample_size=256,
    name='Initial Temperature')
```

为了验证我们创建`val_datset`
```python
val_dataset = hp.data.DecomposedValDataset(
    train_dataset=train_dataset, geometry=geo, size=200, handle=ground, device=args.device)
```
### 定义神经网络和求解器

构建网络和优化器。
```python
network = hp.model.Net(
    hidden_size=args.hidden_size,
    input_sup=torch.tensor([[train_dataset.time_sup, 1., 1., 1.]],
                            device=args.device),
    input_inf=torch.tensor([[train_dataset.time_inf, 0, 0, 0.]],
                            device=args.device), 
    block_num=args.block_num,
    input_dim=4,
    output_dim=1,
    device=args.device)

optimizer = torch.optim.LBFGS( 
    network.parameters(),
    lr=1,
    max_iter=100,  
    tolerance_grad=1e-16,
    tolerance_change=1e-16,
    line_search_fn='strong_wolfe')
```

创建求解器，调用`.solve()`即可开始PINN的流程。

```python
solver = hp.train.TDPINNSolver(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    geometry=geo,
    network=network,
    optimizer=optimizer,
    outer_max_iter=20,
    inner_max_iter=40,
    loss_p=2)

solver.solve()
```