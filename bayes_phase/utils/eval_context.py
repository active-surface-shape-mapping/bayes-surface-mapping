# Following snippet is licensed under MIT license
# Thanks to Christoph Heindl (https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3)

from contextlib import contextmanager


@contextmanager
def eval_context(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
