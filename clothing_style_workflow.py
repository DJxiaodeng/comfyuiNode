"""ComfyUI 服装风格工作流节点
该节点基于ClothingStyleAdapter类，提供完整的工作流功能，使用户可以根据不同国家地区的穿衣风格进行调整
"""

import os
import json
import torch
import numpy as np
from PIL import Image

# 导入服装风格适配器类
from clothing_style_adapter import ClothingStyleAdapter

class ClothingStyleWorkflow:
    """
    服装风格工作流节点，提供完整的工作流功能，使用户可以根据不同国家地区的穿衣风格进行调整
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点输入类型
        """
        # 获取ClothingStyleAdapter支持的所有地区
        adapter = ClothingStyleAdapter()
        regions = adapter.get_all_regions()
        regions.sort()  # 按字母顺序排序地区列表
        
        return {
            "required": {
                "image": ("IMAGE",),
                "region": (regions, {"default": "中国"}),
                "adaptation_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_image_processing": (["是", "否"], {"default": "是"}),
                "workflow_mode": (["完整工作流", "仅提示词调整", "仅图像处理"], {"default": "完整工作流"}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "ddim"], {"default": "euler_ancestral"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STYLE_INFO")
    RETURN_NAMES = ("adapted_image", "adapted_prompt", "adapted_negative_prompt", "seed", "style_info")
    FUNCTION = "run_workflow"
    CATEGORY = "workflow/style"
    
    # 自定义输出类型
    OUTPUT_NODE = True

    def run_workflow(self, image, region, adaptation_strength=0.75, enable_image_processing="是", workflow_mode="完整工作流", 
                   prompt="", negative_prompt="", seed=0, steps=20, cfg=7.0, sampler_name="euler_ancestral",
                   width=512, height=512, batch_size=1):
        """
        运行服装风格工作流
        
        参数:
            image: 输入图像
            region: 目标地区
            adaptation_strength: 调整强度
            enable_image_processing: 是否启用图像处理
            workflow_mode: 工作流模式（完整工作流/仅提示词调整/仅图像处理）
            prompt: 原始提示词
            negative_prompt: 原始负面提示词
            seed: 随机种子
            steps: 采样步数
            cfg: CFG比例
            sampler_name: 采样器名称
            width: 图像宽度
            height: 图像高度
            batch_size: 批处理大小
            
        返回:
            调整后的图像、调整后的提示词、调整后的负面提示词、使用的种子、风格信息
        """
        # 创建服装风格适配器实例
        adapter = ClothingStyleAdapter()
        
        # 根据工作流模式决定处理逻辑
        if workflow_mode == "仅提示词调整":
            # 禁用图像处理
            enable_image_processing = "否"
        elif workflow_mode == "仅图像处理":
            # 保留原始提示词
            pass
        
        # 调用适配器进行风格调整
        adapted_image, adapted_prompt, adapted_negative_prompt, used_seed = adapter.adapt_clothing_style(
            image, region, adaptation_strength, enable_image_processing,
            prompt, negative_prompt, seed, steps, cfg, sampler_name
        )
        
        # 获取地区风格信息
        style_info = adapter.get_style_info(region)
        
        # 创建风格信息字典，包含更多详细信息
        style_info_dict = {
            "region": region,
            "adaptation_strength": adaptation_strength,
            "workflow_mode": workflow_mode,
            "style_info": style_info,
            "original_prompt": prompt,
            "adapted_prompt": adapted_prompt,
            "original_negative_prompt": negative_prompt,
            "adapted_negative_prompt": adapted_negative_prompt,
            "seed": used_seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name
        }
        
        # 输出工作流信息
        self._print_workflow_info(style_info_dict)
        
        # 保存工作流预览（如果需要）
        self._save_workflow_preview(adapted_image, region, used_seed, workflow_mode)
        
        return (adapted_image, adapted_prompt, adapted_negative_prompt, used_seed, style_info_dict)

    def _print_workflow_info(self, style_info):
        """
        打印工作流信息
        
        参数:
            style_info: 风格信息字典
        """
        print("\n===== 服装风格工作流信息 =====")
        print(f"地区: {style_info['region']}")
        print(f"工作流模式: {style_info['workflow_mode']}")
        print(f"调整强度: {style_info['adaptation_strength']}")
        
        region_style = style_info['style_info']
        if region_style:
            print("\n地区风格信息:")
            print(f"推荐风格: {', '.join(region_style.get('推荐风格', []))}")
            print(f"禁忌风格: {', '.join(region_style.get('禁忌风格', []))}")
            print(f"颜色偏好: {', '.join(region_style.get('颜色偏好', []))}")
            print(f"风格描述: {region_style.get('风格描述', '')}")
        
        print("\n提示词信息:")
        print(f"原始提示词: {style_info['original_prompt']}")
        print(f"调整后提示词: {style_info['adapted_prompt']}")
        print(f"原始负面提示词: {style_info['original_negative_prompt']}")
        print(f"调整后负面提示词: {style_info['adapted_negative_prompt']}")
        
        print("\n生成参数:")
        print(f"种子: {style_info['seed']}")
        print(f"步数: {style_info['steps']}")
        print(f"CFG比例: {style_info['cfg']}")
        print(f"采样器: {style_info['sampler_name']}")
        print("==============================\n")

    def _get_output_directory(self):
        """
        获取输出目录路径
        
        返回:
            输出目录的绝对路径
        """
        # 默认输出目录设置在当前工作目录下的'outputs'文件夹
        # 也可以根据需要设置为其他路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "outputs")
        return output_dir
        
    def _save_workflow_preview(self, image, region, seed, workflow_mode):
        """
        保存工作流预览图像
        
        参数:
            image: 处理后的图像
            region: 目标地区
            seed: 使用的种子
            workflow_mode: 工作流模式
        """
        try:
            # 创建预览目录（如果不存在）
            output_dir = self._get_output_directory()
            preview_dir = os.path.join(output_dir, "style_workflow_previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # 保存预览图像
            # 假设image是形状为[1, height, width, 3]的张量
            img_np = image[0].cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # 生成文件名
            filename = f"workflow_{region}_{workflow_mode}_{seed}.png"
            filepath = os.path.join(preview_dir, filename)
            
            # 保存图像
            img_pil.save(filepath)
            print(f"工作流预览已保存至: {filepath}")
        except Exception as e:
            print(f"保存工作流预览时出错: {str(e)}")


class StyleInfoDisplay:
    """
    风格信息显示节点，用于显示服装风格信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点输入类型
        """
        return {
            "required": {
                "style_info": ("STYLE_INFO",),
                "display_mode": (["完整信息", "仅风格信息", "仅提示词信息"], {"default": "完整信息"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style_text",)
    FUNCTION = "display_style_info"
    CATEGORY = "workflow/style"
    
    # 自定义输出类型
    OUTPUT_NODE = True

    def display_style_info(self, style_info, display_mode="完整信息"):
        """
        显示风格信息
        
        参数:
            style_info: 风格信息字典
            display_mode: 显示模式
            
        返回:
            格式化的风格信息文本
        """
        result = []
        
        # 根据显示模式决定显示内容
        if display_mode in ["完整信息", "仅风格信息"]:
            result.append(f"地区: {style_info['region']}")
            
            region_style = style_info['style_info']
            if region_style:
                result.append("\n地区风格信息:")
                result.append(f"推荐风格: {', '.join(region_style.get('推荐风格', []))}")
                result.append(f"禁忌风格: {', '.join(region_style.get('禁忌风格', []))}")
                result.append(f"颜色偏好: {', '.join(region_style.get('颜色偏好', []))}")
                result.append(f"风格描述: {region_style.get('风格描述', '')}")
        
        if display_mode in ["完整信息", "仅提示词信息"]:
            result.append("\n提示词信息:")
            result.append(f"原始提示词: {style_info['original_prompt']}")
            result.append(f"调整后提示词: {style_info['adapted_prompt']}")
            result.append(f"原始负面提示词: {style_info['original_negative_prompt']}")
            result.append(f"调整后负面提示词: {style_info['adapted_negative_prompt']}")
            
            result.append("\n生成参数:")
            result.append(f"种子: {style_info['seed']}")
            result.append(f"步数: {style_info['steps']}")
            result.append(f"CFG比例: {style_info['cfg']}")
            result.append(f"采样器: {style_info['sampler_name']}")
        
        # 将结果连接为字符串
        return ("\n".join(result),)


class RegionalStyleSelector:
    """
    地区风格选择器节点，用于选择特定地区的风格信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点输入类型
        """
        # 获取ClothingStyleAdapter支持的所有地区
        adapter = ClothingStyleAdapter()
        regions = adapter.get_all_regions()
        regions.sort()  # 按字母顺序排序地区列表
        
        return {
            "required": {
                "region": (regions, {"default": "中国"}),
                "output_format": (["提示词", "负面提示词", "风格信息", "全部"], {"default": "提示词"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STYLE_INFO")
    RETURN_NAMES = ("style_prompt", "negative_prompt", "style_info")
    FUNCTION = "select_regional_style"
    CATEGORY = "workflow/style"

    def select_regional_style(self, region, output_format="提示词"):
        """
        选择特定地区的风格信息
        
        参数:
            region: 目标地区
            output_format: 输出格式
            
        返回:
            风格提示词、负面提示词、风格信息
        """
        # 创建服装风格适配器实例
        adapter = ClothingStyleAdapter()
        
        # 获取地区风格信息
        style_info = adapter.get_style_info(region)
        
        # 获取风格提示词和负面提示词
        style_prompt = adapter.STYLE_PROMPT_MAPPING.get(region, "")
        negative_prompt = adapter.TABOO_NEGATIVE_PROMPTS.get(region, "")
        
        # 创建风格信息字典
        style_info_dict = {
            "region": region,
            "style_info": style_info,
            "original_prompt": "",
            "adapted_prompt": style_prompt,
            "original_negative_prompt": "",
            "adapted_negative_prompt": negative_prompt,
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler_ancestral",
            "adaptation_strength": 1.0,
            "workflow_mode": "仅提示词调整"
        }
        
        # 输出选择的风格信息
        print(f"已选择 {region} 地区的风格信息")
        if output_format in ["提示词", "全部"]:
            print(f"风格提示词: {style_prompt}")
        if output_format in ["负面提示词", "全部"]:
            print(f"负面提示词: {negative_prompt}")
        if output_format in ["风格信息", "全部"]:
            print(f"推荐风格: {', '.join(style_info.get('推荐风格', []))}")
            print(f"禁忌风格: {', '.join(style_info.get('禁忌风格', []))}")
            print(f"风格描述: {style_info.get('风格描述', '')}")
        
        return (style_prompt, negative_prompt, style_info_dict)


# 节点列表，用于注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "ClothingStyleWorkflow": ClothingStyleWorkflow,
    "StyleInfoDisplay": StyleInfoDisplay,
    "RegionalStyleSelector": RegionalStyleSelector
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothingStyleWorkflow": "服装风格工作流",
    "StyleInfoDisplay": "风格信息显示器",
    "RegionalStyleSelector": "地区风格选择器"
}