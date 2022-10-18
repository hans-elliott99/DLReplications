import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


def plot_bars(model, grads=False, y_ax_max="max_average"):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    named_paramets = model.named_parameters()\n
    grads = bool. True to plot gradients, false to plot weights.\n
    y_ax_max = 'max_average' or 'mean_average'. how to set the y-axis maximium limit.

    Adapted from: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            layers.append('.'.join(n.split(".")[:-1])) ##remove the ".weight"
            if not grads:
                ave_grads.append(p.data.abs().mean())
                max_grads.append(p.data.abs().max())
                title = 'weight'
            else:
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
                title = 'gradient'

    xticks = range(0,len(ave_grads), 1)
    y_min = min(ave_grads)

    if y_ax_max == "mean_average":
        y_max = (sum(ave_grads)/len(ave_grads))
    else:
        y_max = max(ave_grads)

    for ix in xticks:
        plt.text(ix, y_max/2, s=layers[ix], rotation=90)


    plt.bar(range(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(range(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xticks(range(len(max_grads)))
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = y_min, top=y_max) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel(f"average {title}")
    plt.title(f"{title.capitalize()}s")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], [f'max-{title}', f'mean-{title}', f'zero-{title}'])



def plot_density(model, grads=False, print_info=True, exclude:list=['bias'], start_layer=0, end_layer=None):
    """Plot weights or gradients of layers as density curves to show distributions.  
    """
    if end_layer is None:
        end_layer = len([n for n,p in model.named_parameters()])

    legends = []
    for i, (name, p) in enumerate(model.named_parameters()):
        if any(exc in name for exc in exclude):
            pass
        elif not (start_layer <= i <= end_layer):
            pass
        else:
            if grads:
                tens = p.grad.data
            else:
                tens = p.data

            name = '.'.join(name.split('.')[:-1])
            if print_info:
                print('layer %d (%10s): mean %+f, std %e' % (i, name, tens.mean(), tens.std()))

            hy, hx = torch.histogram(tens, density=True)
            plt.plot(hx[:-1].detach(), hy.detach(), alpha=0.7) ##drop last layer
            legends.append(f'layer {i} ({name})')
    
    plt.legend(legends)


def plot_density_interactive(model, grads=False, print_info=True, exclude:list=['bias'], start_layer=0, end_layer=None):
    """Plot weights or gradients of layers as density curves to show distributions. 
    This function will iterate through all params and generate a plot for each. If running from Jupyter Notebook,
    use a matplotlib backend like TkAgg so plots pop-up.\n
    for example, include %matplotlib tk above the function call.
    """
    if end_layer is None:
        end_layer = len([n for n,p in model.named_parameters()])


    for i, (name, p) in enumerate(model.named_parameters()):
        if any(exc in name for exc in exclude):
            pass
        elif not (int(start_layer) <= int(i) <= int(end_layer)):
            pass
        else:
            if grads:
                tens = p.grad.data
            else:
                tens = p.data

            name = '.'.join(name.split('.')[:-1])
            # print('layer %d (%10s): mean %+f, std %e' % (i, name, tens.mean(), tens.std()))

            plt.figure()
            hy, hx = torch.histogram(tens, density=True)
            plt.title('Layer %d: mean %+f, std %e' % (i, tens.mean(), tens.std()))
            plt.plot(hx[:-1].detach(), hy.detach(), alpha=0.7, label=f'{name}') ##drop last layer
            plt.legend()
            plt.show(block=True)