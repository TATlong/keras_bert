import numpy as np
from bertTAT.bert import AdamWarmup, calc_train_steps


"""
AdamWarmup优化器可用于学习率的「热身」与「衰减」。
学习率将在warmpup_steps步线性增长到lr，并在总共decay_steps步后线性减少到min_lr。
辅助函数calc_train_steps可用于计算这两个步数：
"""

train_x = np.random.standard_normal((1024, 100))

total_steps, warmup_steps = calc_train_steps(
    num_example=train_x.shape[0],
    batch_size=32,
    epochs=10,
    warmup_proportion=0.1,
)
print("total_steps:", total_steps, "\n", "warmup_steps:", warmup_steps)
optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-3, min_lr=1e-5)