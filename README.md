# AI_Medical_System
AI_Medical_System(idea分享，大创/基金项目：AI全自动疾病诊断流程的一些构思）


#-----------------------------------------------#
update on 06/08/2022 by GW.kayak

上大招就：使用LSTM-CLIP时序神经网络结构，可尝试引入深度强化学习！（优点：潜力大，大数据下潜力效果优于机器学习；缺点：模型大，硬件要求高，结构复杂，不易驾驭，需要数据量很大，数据小容易过拟合）
![Imgur](https://github.com/kay-cottage/AI_Medical_System/blob/main/1%20(2).png)


#-----------------------------------------------#
update on 05/08/2022 by GW.kayak


DEMO：baseline主干网络流程图如下：（优点：模型小，硬件较低，需要数据量不大，数据小效果好，容易出结果；缺点：天花板低，大数据时效果比不少深度学习）

![Imgur](https://github.com/kay-cottage/AI_Medical_System/blob/main/1%20(2).png)

#-----------------------------------------------#


update on 05/04/2022 by GW.kayak

类DNS域名解析服务的疾病流程分级解析流程（初筛疾病种类进行分科->对应专科的疾病初筛网络（开出检查项目）->临床决策网络->临床后端评价网络）

初筛网络，决策网络，时序网络，强化学习网络

多个多模态Clip网络作分类器进程集成，深度强化学习state reward的现实交互模式从临床中学习（clip Finetune可以见我另一主页）

LSTM时序网络联系前后网络结果最后做出
