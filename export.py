import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart

from utils import common 
from image_batch import ImageBatcher
from utils.utils_plugin import plugin_NMS

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")



class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
        to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.batch_size = None
        self.network = None
        self.parser = None
        self.workspace = workspace

    def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det, **kwargs):

        '''
        参考： https://github.com/NVIDIA/trt-samples-for-hackathon-cn/blob/master/cookbook/01-SimpleDemo/TensorRT-8.6/main.py
        '''

        # 1、加载 logger 等级, ***** 创建logger：日志记录器
        # available level: VERBOSE, INFO, WARNING, ERROR, INTERNAL_ERROR

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        # if verbose:
        #     self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        # 构建一个序列构建器， build a serialized network from scratch
        self.builder = trt.Builder(self.trt_logger)

        # 1-1、设置网络读取的 flag
        # EXPLICIT_BATCH 相较于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
        # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。
        # 不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # 1-2 构建一个空的网络计算图
        self.network = self.builder.create_network(network_flags)

        # 1-3 create BuidlerConfig to set meta data of the network
        self.config = self.builder.create_builder_config()

        # set workspace for the optimization process (default value is total GPU memory)
        # 2^30 = 1<<30 = 1G
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace * (2 ** 30))

        # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
        # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
        onnx_path = os.path.realpath(onnx_path)
        if not os.path.isfile(onnx_path):
            print("ONNX file not exist. Please check the onnx file path is right ? ")
            return None
        else:
            with open(onnx_path, "rb") as f_r:
                parser_flag = self.parser.parse(f_r.read())
                if not parser_flag:
                    print(f"ERROR: Failed to parse the ONNX file: {onnx_path}")
                    for error in range(self.parser.num_errors):
                        # 出错了，将相关错误的地方打印出来，进行可视化处理
                        print(self.parser.num_errors)
                        print(self.parser.get_error(error))
                    #sys.exit(1)
                    return None

        # 1-6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        print("Network Description")
        for input in inputs:
            # 获取当前转化之前的 输入的 batch_size
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        assert self.batch_size > 0   # 确保 输入的 batch_size 不为零

        v8 = kwargs['v8']
        v10 = kwargs['v10']
        net_work_version = "v5"
        if v8:
            net_work_version = "v8"
        elif v10:
            net_work_version = "v10"
        print(f"now conerrt model: {net_work_version}")

        if end2end:
            self.network = plugin_NMS(self.network, self.trt_logger, max_det, conf_thres, iou_thres,
                                             network_version=net_work_version)

    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        # TODO: Strict type is only needed If the per-layer precision overrides are used
        # If a better method is found to deal with that issue, this flag can be removed.

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                     exact_batches=True))

        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, v8=args.v8, v10=args.v10)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="export the engine include nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.4, type=float,
                        help="The conf threshold for the nms, default: 0.4")
    parser.add_argument("--iou_thres", default=0.5, type=float,
                        help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")
    parser.add_argument("--v8", default=False, action="store_true",
                        help="use yolov8/9 model, default: False")
    parser.add_argument("--v10", default=False, action="store_true",
                        help="use yolov10 model, default: False")
    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)


