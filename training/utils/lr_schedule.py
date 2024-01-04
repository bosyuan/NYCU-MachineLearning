import math
import numpy as np
import matplotlib.pyplot as plt
import os

def cosine_decay(args, batchs: int, result_dir, decay_type: int = 1):
    total_batchs = args.max_epochs * batchs
    # iters = np.arange(total_batchs - args.warmup_batchs)
    iters = np.arange(total_batchs* 0.9)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (args.max_lr - 1e-12) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        # Cyclic cosine decay
        cycles = 4
        cycle_length = int( len(iters) / cycles ) # You can adjust this value based on your preference
        schedule = np.array([0.5 * (args.max_lr - 1e-12) * (cycles - t / cycle_length)/cycles * (1 + math.cos(math.pi * (t % cycle_length) / cycle_length)) for t in iters])
    else:
        raise ValueError("Not support this deccay type")
    
    # if args.warmup_batchs > 0:
    #     warmup_lr_schedule = np.linspace(1e-9, args.max_lr, args.warmup_batchs)
    #     schedule = np.concatenate((warmup_lr_schedule, schedule))

    #set first 10% of iterations to warmup
    warmup_lr_schedule = np.linspace(1e-9, args.max_lr, int(total_batchs/10))
    schedule = np.concatenate((warmup_lr_schedule, schedule))

    plot_schedule(schedule, result_dir)

    return schedule

def exponential_decay(args, batchs: int, result_dir):
    total_batchs = args.max_epochs * batchs
    iters = np.arange(total_batchs*0.9)

    schedule = args.max_lr * np.exp(-0.01 * iters * 100 / batchs)

    warmup_lr_schedule = np.linspace(1e-9, args.max_lr, total_batchs/10)
    schedule = np.concatenate((warmup_lr_schedule, schedule))

    plot_schedule(schedule, result_dir)

    return schedule

def plot_schedule(schedule, result_dir):
    # Plot the learning rate schedule
    plt.plot(schedule)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plot_path = os.path.join(result_dir, 'learning_rate_schedule.png')
    plt.savefig(plot_path)
    plt.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]
