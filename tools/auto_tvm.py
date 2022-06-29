"""
利用auto_tvm优化
"""
import time

import numpy as np
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from PIL import Image


def compile_model(inputs, model_file):
    # load model
    model = torch.jit.load(model_file)

    # input data
    input_names = ["input{}".format(idx) for idx, inp in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    print(f"input_shapes = {input_shapes}")

    # pytorch to relay
    print("=====pytorch to relay start======")
    mod, params = relay.frontend.from_pytorch(model, input_shapes)
    print("====pytorch to relay over====")
    return mod, params


def tune_model(inputs, model_file):
    # compile
    mod, params = compile_model(inputs, model_file)

    # input data
    input_names = ["input{}".format(idx) for idx, inp in enumerate(inputs)]
    input_shapes = list(zip(input_names, [inp.shape for inp in inputs]))
    print(f"input_shapes = {input_shapes}")
    compiled_input = dict(zip(input_names, [inp.clone().cpu().numpy() for inp in inputs]))

    # configurations
    # 测试不同配置的数量
    number = 100
    # 对每个配置进行多少次测量的数量
    repeat = 1
    min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds
    target = "llvm"

    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )

    tuning_option = {
        "tuner": "xgb",
        "trials": 20,
        "early_stopping": 100,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="default"), runner=runner
        ),
        "tuning_records": "resnet-50-v2-autotuning.json",
    }

    # begin by extracting the tasks from the model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank")
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    # check result
    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))


    for name, inp in compiled_input.items():
        module.set_input(name, inp)

    t1 = time.time()
    module.run()
    t2 = time.time()
    print(f"time = {(t2 - t1)*1000}ms")

    result = torch.tensor(module.get_output(0).numpy())
    return result

def img_transforms(img_path):
    """
    图片转换
    :param img_path: string, 图片地址
    :return:
    """
    img = Image.open(img_path).resize((224, 224))

    # Preprocess the image and convert to tensor
    from torchvision import transforms

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = torch.tensor(np.expand_dims(img, 0))
    return img

if __name__ == "__main__":
    # 读取模型
    model_file = "model.pt"

    # 图片转换
    img_path = "cat.jpg"
    img = img_transforms(img_path)

    # 加载并使用
    feature_select_inputs = [img]
    img = tune_model(feature_select_inputs, model_file)





