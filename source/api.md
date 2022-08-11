# HiNPS API 文档

在[HiNPS的仓库](https://github.com/chaoyanggroup/hinps/)中我们将算例均存与[demo](https://github.com/chaoyanggroup/hinps/tree/dev/demo)中，在其中根据不同文件夹的名字可以找到不同的算例而HiNPS的核心代码均存在[hinps](https://github.com/chaoyanggroup/hinps/tree/dev/hinps)目录中。

```{code-block} 
./hinps
├── const.py
├── data
├── distributed
├── expression
├── geometry
├── __init__.py
├── kernels
├── model
├── pdes
├── train
├── unitest
└── utils
```

## 约束

## geometry
## 数据与采样

## 其他

### 日志`logging`

在hinps中我们采用logging包封装了输出信息，并且默认输出模式为logging。
输出由低到高分为如下表。级别低于输出模式的日志都会被输出。层级越高的日志输出应当越少。

```{table}
align center
widths grid

level    | hinps API   | 含义
critical | `hinps.critical`| 严重错误、必须停止执行时的信息
error    | `hinps.error`   | 因为其他地方出现严重问题导致某处出错，不能正常运行
warning  | `hinps.warning` | 可能出错、使用旧代码等警告
info     | `hinps.info`    | 普通信息，可用于观察程序运行结果符合预期
debug    | `hinps.debug`   | 调试信息，可用于展示大量细节信息
```

在hinps中为了适应分布式的情况，我们默认每次日志打印都由Rank 0，也就是主进程来打印。如果需要检查每个进程的情况，需要给出参数`force=True`。

### 符号表达式与计算
### 计时