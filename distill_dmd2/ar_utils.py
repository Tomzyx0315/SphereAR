import torch.distributed as dist

from SphereAR.utils import requires_grad


AR_BACKBONE_MODULES = (
    "cls_embedding",
    "proj_in",
    "emb_norm",
    "layers",
    "norm",
    "pos_for_diff",
)


def ar_backbone_modules(model):
    return [(name, getattr(model, name)) for name in AR_BACKBONE_MODULES]


def ar_backbone_parameters(model):
    for _name, module in ar_backbone_modules(model):
        yield from module.parameters()


def set_module_requires_grad(module, flag):
    for param in module.parameters():
        param.requires_grad_(flag)


def set_ar_backbone_trainable(teacher, enabled):
    requires_grad(teacher, False)
    teacher.eval()
    teacher.vae.eval()
    teacher.head.eval()
    if enabled:
        for _name, module in ar_backbone_modules(teacher):
            module.train()
            set_module_requires_grad(module, True)
        teacher.vae.eval()
        teacher.head.eval()


def ar_backbone_state_dict(teacher):
    state = {}
    for name, module in ar_backbone_modules(teacher):
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value
    return state


def load_ar_backbone_state_dict(teacher, state):
    for name, module in ar_backbone_modules(teacher):
        prefix = f"{name}."
        module_state = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        module.load_state_dict(module_state, strict=True)


def all_reduce_trainable_grads(params):
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    for param in params:
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
