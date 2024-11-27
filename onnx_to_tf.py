
import argparse
import onnx
from onnx_tf.backend import prepare
import traceback

def onnx_to_tensorflow(onnx_model_path, output_dir):
    """
    将 ONNX 模型转换为 TensorFlow SavedModel 格式，并输出详细的调试信息。
    """
    try:
        print(f"尝试加载 ONNX 模型: {onnx_model_path}")
        # 检查 ONNX 模型是否存在并加载
        onnx_model = onnx.load(onnx_model_path)
        print("ONNX 模型加载成功！")

        print("验证 ONNX 模型...")
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证成功！")

        # 转换为 TensorFlow 格式
        print("开始转换 ONNX 模型为 TensorFlow 格式...")
        tf_rep = prepare(onnx_model)
        print("转换成功！")

        # 保存为 TensorFlow SavedModel 格式
        print(f"保存 TensorFlow 模型到: {output_dir}")
        tf_rep.export_graph(output_dir)
        print("模型转换完成，保存成功！")

    except Exception as e:
        print("模型转换过程中发生错误！")
        print("错误信息如下：")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 ONNX 模型转换为 TensorFlow SavedModel 格式")
    parser.add_argument("onnx_model_path", type=str, help="输入的 ONNX 模型文件路径")
    parser.add_argument("output_dir", type=str, help="输出的 TensorFlow SavedModel 保存目录")

    args = parser.parse_args()
    print("开始模型转换任务...")
    onnx_to_tensorflow(args.onnx_model_path, args.output_dir)
    print("任务结束。")
