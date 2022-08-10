.. HiNPS documentation master file, created by
   sphinx-quickstart on Mon Aug  8 16:34:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HiNPS文档
=================================

HiNPS, High-performance Neural Network PDE Solver, 是一个基于神经网络方法求解偏微分方程的软件。

   几何区域
      解析定义区域
      STL文件定义区域
      区域分解
   方程与约束
      偏微分方程
      第一、二、三类边界条件
      交换边界信息
   数据模块
      数据的组织
      采样
   分布式模块
      映射
      通信源语

```{admonition} 友情提示
   本项目依然在活跃开发中。This project is under active developtment.
```

```{eval-rst}
.. toctree::
   :maxdepth: 2

   快速入门<get_started.md>
   方程与约束<constraint.md>
   几何区域<geometry.md>
   数据与采样<data.md>
   并行与分布式<distributed.md>
```

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: API文档

   hinps<api.md>
   hinps.train<api.md>
```


<!-- 
.. Indices and tables
.. ==================
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search` -->
