import numpy as np
import tensorrt as trt

def plugin_NMS(network, trt_logger, max_det, conf_thres, iou_thres, network_version='v5'):

    # 添加NMS的步骤，按照正常的神经网络的搭建方式将NMS 构建为一层，添加进去自己的网络层当中

    # 需要先将之前标签为 “输出” 的输出标签拿掉，后续根据NMS插件的输出再自己加
    # 之前的输出，后续会作为某一添加层的输入，后面会看到
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    try:
        for previous_output in outputs:
            network.unmark_output(previous_output)
    except:
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)

    # 拆分得到的输出，将其满足 EfficientNMS_TRT 的输入的要求
    # 根据不同 YOLO 版本NMS添加方式不同
    if network_version == "v8":
        '''
        output [1, 84, 8400]
        原 YOLO—V8 的输出 为 1 * 8400 * 84 -> [batch, box_nums, box + cls_num]
        拆分自己得到的输出，将其满足 EfficientNMS_TRT 的输入的要求
        '''
        strides = trt.Dims([1, 1, 1])
        starts = trt.Dims([0, 0, 0])

        '''
        添加一个 Reshape层，将原来的输出 previous_output 作为下一个新加层（Reshape层）的输入，
        Reshape 输出进行，维度变换 [1, 84, 8400] -> [1, 8400, 84]，达到重塑输入的目的
        '''
        previous_output = network.add_shuffle(previous_output)
        previous_output.second_transpose = (0, 2, 1)
        # output [1, 8400, 84]
        bs, num_boxes, temp = previous_output.get_output(0).shape
        shapes = trt.Dims([bs, num_boxes, 4])

        '''
        对 [1, 8400, 84] 的输入 从 [0, 0, 0] 开始，按照 [1, num_boxes , 4] 以 [1, 1, 1] 的步幅大小进行切片操作 
        得到 [1, num_boxes, 4]
        '''
        boxes = network.add_slice(previous_output.get_output(0), starts, shapes, strides)
        num_classes = temp - 4  # 前4个是box坐标， 得到类别数
        starts[2] = 4
        shapes[2] = num_classes

        '''
        对 [1, 8400, 84] 的输入 从 [0, 0, 4] 位置开始，按照 [1, num_boxes , 80] 以 [1, 1, 1] 的步幅大小进行切片操作 
        得到 [1, 8400, 80], 每个box对应的score
        '''
        scores = network.add_slice(previous_output.get_output(0), starts, shapes, strides)

    elif network_version == "v5":
        # output [1, 8400, 85] -> [batch, box_nums, box + obj_score + class_scores]
        strides = trt.Dims([1, 1, 1])
        starts = trt.Dims([0, 0, 0])
        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        # start:[0, 0, 0] shapes:[1, 8400, 4] strides:[1, 1, 1] ->  结果：[1, 8400, 4]
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        num_classes = temp - 5
        starts[2] = 4
        shapes[2] = 1
        # 得到每个类别score， [0, 0, 4] [1, 8400, 1] [1, 1, 1]  -> [1, 8400, 4]
        class_scores = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        # 得到存在目标的score， [0, 0, 5] [1, 8400, 80] [1, 1, 1] -> [1, 8400, 1]
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        # 再重新计算分数, scores = obj_score * class_scores => [bs, num_boxes, nc]
        scores = network.add_elementwise(class_scores.get_output(0), obj_score.get_output(0), trt.ElementWiseOperation.PROD)

    elif network_version == "v10":  # 没有测试过

        # output [1, 300, 6]
        # 添加 TopK 层，在第二个维度上找到前 100 个最大值 [1, 100, 6]
        strides = trt.Dims([1, 1, 1])
        starts = trt.Dims([0, 0, 0])
        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 4
        shapes[2] = 1
        # [0, 0, 4] [1, 300, 1] [1, 1, 1]
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        # [0, 0, 5] [1, 300, 1] [1, 1, 1]
        cls = network.add_slice(previous_output, starts, shapes, strides)
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        print("YOLOv10 Modify")

        def squeeze(previous_output):
            reshape_dims = (bs, 300)
            previous_output = network.add_shuffle(previous_output.get_output(0))
            previous_output.reshape_dims = reshape_dims
            return previous_output

        # 定义常量值和形状
        constant_value = 300.0
        constant_shape = (300,)
        constant_data = np.full(constant_shape, constant_value, dtype=np.float32)
        num = network.add_constant(constant_shape, trt.Weights(constant_data))
        num.get_output(0).name = "num"
        network.mark_output(num.get_output(0))
        boxes.get_output(0).name = "boxes"
        network.mark_output(boxes.get_output(0))
        obj_score = squeeze(obj_score)
        obj_score.get_output(0).name = "scores"
        network.mark_output(obj_score.get_output(0))
        cls = squeeze(cls)
        cls.get_output(0).name = "classes"
        network.mark_output(cls.get_output(0))

        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    '''
    初始化注册插件库
    使用可选名称空间将所有现有 TensorRT 插件初始化并注册到 IPluginRegistry。
    加载TensorRT的插件库，以便可以使用插件API创建和配置自定义层。
    '''
    trt.init_libnvinfer_plugins(trt_logger, namespace="")

    # 在当前的 runtime 的 环境下，返回 plugin 的 register
    registry = trt.get_plugin_registry()
    assert (registry)

    '''
    根据输入的 plugin 的 plugin_namsspace, type_version 返回一个 plugin creator
    EfficientNMS_TRT 是在 tensorrt 中已经写的 layer
    '''
    creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
    assert (creator)

    '''
    tensorrt 中 EfficientNMS_TRT 版本为 1 的输入的要求如下
    
    "plugin_version": "1",
    "background_class": -1,  # no background class
    "max_output_boxes": max_det_per_img, # 每张图片最多检测数
    "score_threshold": score_thresh, 
    "iou_threshold": nms_thresh,
    "score_activation": False,
    "box_coding": 1,
    '''

    # 将一些必要的参数 trt 化，并后面传入 EfficientNMS_TRT 中
    fc = []
    fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("score_activation", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))

    fc = trt.PluginFieldCollection(fc)
    nms_layer = creator.create_plugin("nms_layer", fc)

    '''
    将插件 NMS 节点添加到网络
    通过add_plugin_v2直接调用这里面已有的Plugin ,
    将NMS Plugin 插入到Network中 add_plugin_v2([inputT0], plugin)
    [inputT0] 需要满足所使用插件的输入要求这里是[boxes， scores]
    '''
    layer = network.add_plugin_v2([boxes.get_output(0), scores.get_output(0)], nms_layer)

    # 插件 NMS Plugin 的输出，重新定义模型的输出
    layer.get_output(0).name = "num"
    layer.get_output(1).name = "boxes"
    layer.get_output(2).name = "scores"
    layer.get_output(3).name = "classes"
    for i in range(4):
        network.mark_output(layer.get_output(i))
    return network



if __name__ == "__main__":
    pass