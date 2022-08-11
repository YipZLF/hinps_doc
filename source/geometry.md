# 几何区域

偏微分方程都是定义在一定空间区域上，本模块的功能是表示和管理空间区域。边界条件，也就是边界上的约束，也是偏微分方程的一个重要部分，所以我们还需要维护空间区域的边界。

整体来看，本模块的实现逻辑是：

- 描述几何图形的核心是SDF, Signed Distance Function。

```{admonition} SDF的定义
对于一个封闭几何区域 {math}`\Omega \in \mathcal{R}^d`, 我们构造一个函数{math}`F: \mathcal{R}^d \rightarrow \mathcal{R}`。满足：

- 如果{math}`F(\mathbf{x})>0`，则{math}`\mathbf{x}\in \Omega / \partial\Omega`，{math}`\mathbf{x}`在{math}`\Omega`内
- 如果{math}`F(\mathbf{x})<0`，则{math}`\mathbf{x} \in \overline{\Omega}`，{math}`\mathbf{x}`在{math}`\Omega`外
- 如果{math}`F(\mathbf{x})=0`，则{math}`\mathbf{x} \in \partial \Omega`，{math}`\mathbf{x}`在{math}`\Omega`边界上。

我们称F是 {math}`\Omega`的一个SDF。
```
    - 例如，对于球来说，SDF可以定义为其到球心距离减去半径；
- `Geometry`需要维护采样空间(sample space)，提供给[`Sampler`](data.md#sampler采样器)用于采样。
   
- 每个几何图形都维护了边界`Boundary`的信息。边界本质上是低一个维度的几何图形，需要维护其法向信息，其他大体逻辑与`Geometry`相似。


## `Geometry` 几何区域

我们以d维的`Box`为例子，介绍我们如何表示一个有解析表达式的几何区域。

### 定义形状

形状用SDF来表示。`sdf`方法接受的输入是采样点的数据Tensor，大小必须为[...,d]，即最后一维长度为d。返回的数据大小为[...,1]。

我们需要把这个几何区域的解析表达式写成一个函数句柄，保存为`self._sdf`，从而给`sdf`方法来调用。对于`Box`来说，需要的参数就是每个维度可取值的上界`sup`和下界`inf`。

```{admonition} 注意
考虑到浮点数精度，我们在代码中不使用SDF(x)==0这一判断，而是使用abs(SDF(x))< EPS。
```

### 定义边界`Boundary`

我们需要创建边界对象：给出边界需要的参数，具体见[`Boundary`](#boundary边界)。然后保存到`self._bd`中。

这一部分的目的是采样的时候我们可以遍历每个边界，并在其上进行采样。

为了采样合理，我们用MC方法估算了每个边界的“面积”，用于分配在不同边界上面的采样数目。

### 原空间与采样空间的映射

当原空间（origin_space，定义PDE的空间）和采样空间不一致时，`Geometry`还需要维护这两空间之间的映射。例如：对于平面直角坐标系中的圆，其空间$\Omega = \{x| \|x\|_2 \le r \}$；我们可以将其转换为极坐标，定义其采样空间为$[0,r]\times[0,2\pi]$

`Box`的可以不需要这样的映射，所以
- `to_sample_space`是直接返回上界和下界
- `to_origin_space`是一个恒等映射

`filter_sample`这一方法的目的是，我们允许原空间到采样空间的映射不是一一对应的，为了简化代码的实现
- 例如：我们可以把一个半径为r的圆映射到$[-r,r] \times [-r,r]$，如果我们在采样空间进行均匀采样后，得到的点只有$\frac{\pi}{4}$会落在原来的圆内。

我们在`filter_sample`中调用`sdf`可以将不在圆内的点筛掉。如果输入是一个尺寸为[...,N,d]的数据Tensor，返回则是一个尺寸为[...,M,d]的Tensor，其中M $\le$ N。

 
```{admonition} 提示
在采样空间的分布与原空间的分布不一定是一样的，以上述圆为例子，在采样空间的均匀采样并不是在原空间的均匀采样。

相关加权的采样逻辑需要在`Sampler`中实现。
```
## `Boundary`边界

我们定义边界形状与`Geometry`略有不同，另外需要定义边界法向。
### 定义形状

我们要求边界都是可参数化的，需要给出参数范围和参数的映射函数（只需用`SymPy`给出数学表达式，使用`hinps.expression.Function`转换为可调用的Python函数）。

- 如果是一个不与坐标轴平行的正方形ABCD，我们可以用$(a,b) \in [0,1]\times[0,1]$作为参数，通过映射$g(a,b)=a * \overrightarrow{AB} + b * \overrightarrow{AD}$得到这个正方形内的点。


### 原空间与采样空间的映射

采样空间就是该边界的参数空间。

### 定义法向
我们需要在定义边界的时候给出法向的函数，他是一个可调用的Python函数。输入为[...,d]，输出也为[...,d]，而且输出的每一个法向应该是单位向量。

```{admonition} 注意
我们维护的是**外法向**。
```