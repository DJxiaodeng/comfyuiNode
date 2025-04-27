"""ComfyUI 图片信息查看节点
可以查看图片的元数据信息，包括标签、种子、Lora等
支持图像预览、颜色分析和批量处理
"""

import os
import json
import re
import io
import numpy as np
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont
import torch
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any

class ImageInfoExtractor:
    """
    图片信息提取器，用于从图片中提取元数据
    支持图像预览、颜色分析和批量处理
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "analysis_mode": (["基本信息", "详细信息", "全部信息"], {"default": "全部信息"}),
            },
            "optional": {
                "save_to_file": ("BOOLEAN", {"default": False}),
                "output_path": ("STRING", {"default": "image_info.txt"}),
                "generate_preview": ("BOOLEAN", {"default": False}),
                "analyze_colors": ("BOOLEAN", {"default": False}),
                "color_sample_size": ("INT", {"default": 1000, "min": 100, "max": 10000}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE",)
    RETURN_NAMES = ("info", "preview_image")
    FUNCTION = "extract_info"
    CATEGORY = "image/info"
    
    def extract_info(self, 
                    image: torch.Tensor, 
                    analysis_mode: str = "全部信息",
                    save_to_file: bool = False, 
                    output_path: str = "image_info.txt",
                    generate_preview: bool = False,
                    analyze_colors: bool = False,
                    color_sample_size: int = 1000) -> Tuple[str, torch.Tensor]:
        """
        从图片中提取元数据信息
        
        参数:
            image: 输入图像（可以是单张图像或批量图像）
            analysis_mode: 分析模式（基本信息、详细信息、全部信息）
            save_to_file: 是否将信息保存到文件
            output_path: 保存信息的文件路径
            generate_preview: 是否生成预览图像
            analyze_colors: 是否分析图像颜色
            color_sample_size: 颜色采样大小
            
        返回:
            提取的信息文本和预览图像
        """
        # 检查是否为批量图像
        batch_mode = len(image.shape) == 4 and image.shape[0] > 1
        result_info = []
        preview_images = []
        
        # 处理批量图像或单张图像
        if batch_mode:
            result_info.append(f"## 批量处理 {image.shape[0]} 张图像")
            result_info.append("")
            
            for i in range(image.shape[0]):
                single_image = image[i:i+1]
                img_pil = self._tensor_to_pil(single_image)
                
                result_info.append(f"### 图像 {i+1}")
                info, preview = self._process_single_image(
                    img_pil, 
                    analysis_mode, 
                    analyze_colors,
                    color_sample_size,
                    generate_preview
                )
                result_info.append(info)
                
                if generate_preview:
                    preview_images.append(preview)
        else:
            # 单张图像处理
            img_pil = self._tensor_to_pil(image)
            info, preview = self._process_single_image(
                img_pil, 
                analysis_mode, 
                analyze_colors,
                color_sample_size,
                generate_preview
            )
            result_info.append(info)
            
            if generate_preview:
                preview_images.append(preview)
        
        # 合并结果
        final_info = "\n".join(result_info)
        
        # 如果需要，保存到文件
        if save_to_file:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(final_info)
                print(f"图片信息已保存到: {output_path}")
            except Exception as e:
                print(f"保存文件时出错: {str(e)}")
        
        # 准备返回的预览图像
        if generate_preview and preview_images:
            if batch_mode:
                # 将多个预览图像拼接成一个网格
                preview_tensor = self._create_image_grid(preview_images)
            else:
                # 单张图像直接转换为tensor
                preview_tensor = self._pil_to_tensor(preview_images[0])
        else:
            # 如果没有生成预览，返回原始图像
            preview_tensor = image if not batch_mode else image[0:1]
            
        return (final_info, preview_tensor)
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """
        将tensor转换为PIL图像
        
        参数:
            tensor: 输入tensor
            
        返回:
            PIL图像对象
        """
        i = 255. * tensor.cpu().numpy()
        return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)[0])
    
    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """
        将PIL图像转换为tensor
        
        参数:
            img: PIL图像对象
            
        返回:
            图像tensor
        """
        image_np = np.array(img).astype(np.float32) / 255.0
        # 确保图像是3通道的
        if len(image_np.shape) == 2:  # 灰度图像
            image_np = np.stack([image_np, image_np, image_np], axis=2)
        elif image_np.shape[2] == 4:  # RGBA图像
            image_np = image_np[:, :, :3]
        
        # 转换为CHW格式并添加批次维度
        image_np = np.transpose(image_np, (2, 0, 1))
        return torch.from_numpy(image_np).unsqueeze(0)
    
    def _create_image_grid(self, images: List[Image.Image]) -> torch.Tensor:
        """
        将多个图像拼接成网格
        
        参数:
            images: 图像列表
            
        返回:
            网格图像tensor
        """
        # 确定网格大小
        num_images = len(images)
        cols = min(4, num_images)  # 最多4列
        rows = (num_images + cols - 1) // cols
        
        # 找到最大宽度和高度
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # 创建网格图像
        grid_width = cols * max_width
        grid_height = rows * max_height
        grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # 放置图像
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * max_width
            y = row * max_height
            grid_img.paste(img, (x, y))
        
        return self._pil_to_tensor(grid_img)
    
    def _process_single_image(self, 
                             img: Image.Image, 
                             analysis_mode: str,
                             analyze_colors: bool,
                             color_sample_size: int,
                             generate_preview: bool) -> Tuple[str, Optional[Image.Image]]:
        """
        处理单张图像
        
        参数:
            img: PIL图像对象
            analysis_mode: 分析模式
            analyze_colors: 是否分析颜色
            color_sample_size: 颜色采样大小
            generate_preview: 是否生成预览
            
        返回:
            信息文本和预览图像
        """
        result = []
        
        # 提取基本信息
        result.append("## 图片基本信息")
        result.append(f"尺寸: {img.width} x {img.height}")
        result.append(f"格式: {img.format or '未知'}")
        result.append(f"模式: {img.mode}")
        result.append("")
        
        # 如果只需要基本信息，则返回
        if analysis_mode == "基本信息":
            info = "\n".join(result)
            preview = self._generate_preview(img, info) if generate_preview else None
            return info, preview
        
        # 提取PNG信息
        metadata = {}
        if hasattr(img, 'info') and img.info:
            result.append("## PNG元数据")
            
            # 提取常见的ComfyUI/SD元数据
            metadata = self._parse_png_metadata(img)
            
            # 提取生成参数
            if "parameters" in metadata:
                result.append("### 生成参数")
                result.append(metadata["parameters"])
                result.append("")
            
            # 提取提示词
            if "prompt" in metadata:
                result.append("### 正面提示词")
                result.append(metadata["prompt"])
                result.append("")
            
            if "negative_prompt" in metadata:
                result.append("### 负面提示词")
                result.append(metadata["negative_prompt"])
                result.append("")
            
            # 提取种子
            if "seed" in metadata:
                result.append(f"种子: {metadata['seed']}")
            
            # 提取采样器和步数
            if "sampler" in metadata:
                result.append(f"采样器: {metadata['sampler']}")
            if "steps" in metadata:
                result.append(f"步数: {metadata['steps']}")
            
            # 提取CFG值
            if "cfg" in metadata or "cfg_scale" in metadata:
                cfg = metadata.get("cfg", metadata.get("cfg_scale", "未知"))
                result.append(f"CFG值: {cfg}")
            
            # 提取模型信息
            if "model" in metadata or "model_hash" in metadata:
                model = metadata.get("model", "未知")
                model_hash = metadata.get("model_hash", "")
                result.append(f"模型: {model} ({model_hash})")
            
            # 提取Lora信息
            loras = self._extract_loras(metadata)
            if loras:
                result.append("\n### 使用的Lora")
                for lora in loras:
                    result.append(f"- {lora['name']} (权重: {lora['weight']})")
            
            # 如果只需要详细信息，则返回
            if analysis_mode == "详细信息":
                info = "\n".join(result)
                preview = self._generate_preview(img, info) if generate_preview else None
                return info, preview
            
            # 提取其他元数据（全部信息模式）
            result.append("\n### 其他元数据")
            for key, value in metadata.items():
                if key not in ["parameters", "prompt", "negative_prompt", "seed", 
                              "sampler", "steps", "cfg", "cfg_scale", "model", "model_hash"]:
                    # 跳过已经处理过的键和Lora相关键
                    if not any(key.startswith(prefix) for prefix in ["lora_", "ti_", "lyco_"]):
                        result.append(f"{key}: {value}")
        
        # 颜色分析（如果启用）
        if analyze_colors:
            color_info = self._analyze_colors(img, color_sample_size)
            result.append("\n## 颜色分析")
            result.append(f"主色调: {color_info['dominant_color']}")
            result.append("\n### 颜色分布 (RGB, 出现频率)")
            for color, freq in color_info['color_distribution']:
                result.append(f"- RGB{color}: {freq:.2f}%")
        
        # 合并结果
        info = "\n".join(result)
        
        # 生成预览图像（如果启用）
        preview = self._generate_preview(img, info) if generate_preview else None
        
        return info, preview
    
    def _parse_png_metadata(self, img: Image.Image) -> Dict[str, Any]:
        """
        解析PNG图像的元数据
        
        参数:
            img: PIL图像对象
            
        返回:
            解析后的元数据字典
        """
        metadata = {}
        
        # 提取PNG文本块
        if hasattr(img, 'text') and img.text:
            for key, value in img.text.items():
                metadata[key] = value
        
        # 尝试从parameters字段解析更多信息
        if "parameters" in metadata:
            params = metadata["parameters"]
            
            # 提取提示词
            prompt_match = re.search(r"^(.*?)(?:Negative prompt:|$)", params, re.DOTALL)
            if prompt_match:
                metadata["prompt"] = prompt_match.group(1).strip()
            
            # 提取负面提示词
            neg_prompt_match = re.search(r"Negative prompt:(.*?)(?:Steps:|$)", params, re.DOTALL)
            if neg_prompt_match:
                metadata["negative_prompt"] = neg_prompt_match.group(1).strip()
            
            # 提取种子
            seed_match = re.search(r"Seed: (\d+)", params)
            if seed_match:
                metadata["seed"] = seed_match.group(1)
            
            # 提取采样器
            sampler_match = re.search(r"Sampler: ([^,]+)", params)
            if sampler_match:
                metadata["sampler"] = sampler_match.group(1).strip()
            
            # 提取步数
            steps_match = re.search(r"Steps: (\d+)", params)
            if steps_match:
                metadata["steps"] = steps_match.group(1)
            
            # 提取CFG值
            cfg_match = re.search(r"CFG scale: ([\d.]+)", params)
            if cfg_match:
                metadata["cfg_scale"] = cfg_match.group(1)
            
            # 提取模型信息
            model_match = re.search(r"Model: ([^,]+)", params)
            if model_match:
                metadata["model"] = model_match.group(1).strip()
            
            # 提取模型哈希
            model_hash_match = re.search(r"Model hash: ([^,]+)", params)
            if model_hash_match:
                metadata["model_hash"] = model_hash_match.group(1).strip()
        
        # 尝试从ComfyUI元数据中提取信息
        if "ComfyUI" in metadata:
            try:
                comfy_data = json.loads(metadata["ComfyUI"])
                if isinstance(comfy_data, dict) and "prompt" in comfy_data:
                    prompt = comfy_data["prompt"]
                    
                    # 从ComfyUI工作流中提取更多信息
                    for node_id, node in prompt.items():
                        if "inputs" in node:
                            inputs = node["inputs"]
                            
                            # 提取种子
                            if "seed" in inputs:
                                metadata["seed"] = inputs["seed"]
                            
                            # 提取CFG值
                            if "cfg" in inputs:
                                metadata["cfg"] = inputs["cfg"]
                            
                            # 提取步数
                            if "steps" in inputs:
                                metadata["steps"] = inputs["steps"]
                            
                            # 提取采样器
                            if "sampler_name" in inputs:
                                metadata["sampler"] = inputs["sampler_name"]
                            
                            # 提取提示词
                            if "text" in inputs and node["class_type"] in ["CLIPTextEncode", "KSampler"]:
                                if "positive" in node_id.lower():
                                    metadata["prompt"] = inputs["text"]
                                elif "negative" in node_id.lower():
                                    metadata["negative_prompt"] = inputs["text"]
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"解析ComfyUI元数据时出错: {str(e)}")
        
        # 尝试解析EXIF数据
        try:
            if hasattr(img, '_getexif') and callable(img._getexif):
                exif = img._getexif()
                if exif:
                    metadata["exif"] = {}
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        metadata["exif"][tag_name] = str(value)
        except Exception:
            # EXIF解析错误，忽略
            pass
        
        return metadata
    
    def _extract_loras(self, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        从元数据中提取Lora信息
        
        参数:
            metadata: 图像元数据字典
            
        返回:
            Lora信息列表
        """
        loras = []
        
        # 从参数文本中提取Lora信息
        if "parameters" in metadata:
            lora_pattern = r"<lora:([^:]+):([^>]+)>"
            for match in re.finditer(lora_pattern, metadata["parameters"]):
                lora_name = match.group(1)
                lora_weight = match.group(2)
                loras.append({"name": lora_name, "weight": lora_weight})
        
        # 从ComfyUI元数据中提取Lora信息
        for key, value in metadata.items():
            if key.startswith("lora_"):
                lora_name = key.replace("lora_", "")
                loras.append({"name": lora_name, "weight": value})
            # 检查lycoris/hypernetwork
            elif key.startswith("lyco_") or key.startswith("ti_"):
                lora_type = "LyCORIS" if key.startswith("lyco_") else "Textual Inversion"
                lora_name = key.replace("lyco_", "").replace("ti_", "")
                loras.append({"name": f"{lora_name} ({lora_type})", "weight": value})
        
        return loras
    
    def _analyze_colors(self, img: Image.Image, sample_size: int = 1000) -> Dict[str, Any]:
        """
        分析图像的颜色分布
        
        参数:
            img: PIL图像对象
            sample_size: 采样大小
            
        返回:
            颜色分析结果
        """
        # 转换为RGB模式
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 缩小图像以加快处理速度
        width, height = img.size
        scale = min(1.0, np.sqrt(sample_size / (width * height)))
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_small = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            img_small = img
        
        # 获取像素数据
        pixels = list(img_small.getdata())
        
        # 计算颜色频率
        color_counter = Counter(pixels)
        total_pixels = len(pixels)
        
        # 获取主色调
        dominant_color = color_counter.most_common(1)[0][0]
        
        # 获取颜色分布（前10种颜色）
        color_distribution = [
            (color, count / total_pixels * 100)
            for color, count in color_counter.most_common(10)
        ]
        
        return {
            "dominant_color": f"RGB{dominant_color}",
            "color_distribution": color_distribution
        }
    
    def _generate_preview(self, img: Image.Image, info: str) -> Image.Image:
        """
        生成带有信息的预览图像
        
        参数:
            img: 原始图像
            info: 图像信息文本
            
        返回:
            预览图像
        """
        # 计算预览图像大小
        max_width = 1024
        max_height = 1024
        
        # 缩放原始图像
        width, height = img.size
        scale = min(max_width / width, max_height / height, 1.0)
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        else:
            img_resized = img.copy()
            new_width, new_height = width, height
        
        # 创建信息面板
        info_panel_height = min(new_height, 400)  # 信息面板高度
        info_panel_width = new_width
        
        # 创建预览图像（图像 + 信息面板）
        preview_width = new_width
        preview_height = new_height + info_panel_height
        preview = Image.new('RGB', (preview_width, preview_height), color=(240, 240, 240))
        
        # 粘贴缩放后的图像
        preview.paste(img_resized, (0, 0))
        
        # 在信息面板上绘制文本
        draw = ImageDraw.Draw(preview)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            # 尝试使用系统字体
            font_path = "C:\\Windows\\Fonts\\simhei.ttf"  # Windows中文字体
            font = ImageFont.truetype(font_path, 14)
        except IOError:
            # 使用默认字体
            font = ImageFont.load_default()
        
        # 提取关键信息用于预览
        key_info = self._extract_key_info(info)
        
        # 绘制信息
        text_y = new_height + 10
        for line in key_info.split('\n'):
            draw.text((10, text_y), line, fill=(0, 0, 0), font=font)
            text_y += 20
            if text_y > preview_height - 10:
                break
        
        return preview
    
    def _extract_key_info(self, info: str) -> str:
        """
        从完整信息中提取关键信息用于预览
        
        参数:
            info: 完整信息文本
            
        返回:
            关键信息文本
        """
        key_info_lines = []
        
        # 提取尺寸信息
        size_match = re.search(r"尺寸: (\d+ x \d+)", info)
        if size_match:
            key_info_lines.append(f"尺寸: {size_match.group(1)}")
        
        # 提取种子信息
        seed_match = re.search(r"种子: (\d+)", info)
        if seed_match:
            key_info_lines.append(f"种子: {seed_match.group(1)}")
        
        # 提取模型信息
        model_match = re.search(r"模型: ([^\n]+)", info)
        if model_match:
            key_info_lines.append(f"模型: {model_match.group(1)}")
        
        # 提取采样器和步数
        sampler_match = re.search(r"采样器: ([^\n]+)", info)
        if sampler_match:
            key_info_lines.append(f"采样器: {sampler_match.group(1)}")
        
        steps_match = re.search(r"步数: (\d+)", info)
        if steps_match:
            key_info_lines.append(f"步数: {steps_match.group(1)}")
        
        # 提取CFG值
        cfg_match = re.search(r"CFG值: ([^\n]+)", info)
        if cfg_match:
            key_info_lines.append(f"CFG值: {cfg_match.group(1)}")
        
        # 如果有主色调信息
        color_match = re.search(r"主色调: ([^\n]+)", info)
        if color_match:
            key_info_lines.append(f"主色调: {color_match.group(1)}")
        
        # 添加提示信息
        key_info_lines.append("(查看完整信息请参阅输出文本)")
        
        return "\n".join(key_info_lines)

# 尝试导入EXIF标签
try:
    from PIL.ExifTags import TAGS
except ImportError:
    # 如果导入失败，创建一个空字典
    TAGS = {}

# 节点列表，用于注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "ImageInfoExtractor": ImageInfoExtractor,
    "BatchImageInfoExtractor": ImageInfoExtractor  # 兼容性别名
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfoExtractor": "图片信息查看器",
    "BatchImageInfoExtractor": "批量图片信息查看器"
}