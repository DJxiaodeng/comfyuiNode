"""
ComfyUI 服装风格自适应节点
可以根据不同国家和地区的文化习惯自动调整服装风格，避开当地禁忌风格搭配
"""

import os
import json
import re
import numpy as np
from PIL import Image
import torch
from pathlib import Path

class ClothingStyleAdapter:
    """
    服装风格自适应节点，用于根据不同国家和地区的文化习惯自动调整服装风格
    """
    
    # 国家/地区服装风格数据库
    STYLE_DATABASE = {
        "中国": {
            "推荐风格": ["传统汉服", "中式改良服装", "现代中式元素", "旗袍", "唐装"],
            "禁忌风格": ["过于暴露", "带有不尊重传统文化的图案"],
            "颜色偏好": ["红色", "金色", "蓝色", "黄色"],
            "风格描述": "中国传统服饰强调含蓄、典雅，现代服饰融合传统元素与现代设计。"
        },
        "日本": {
            "推荐风格": ["和服", "改良和服", "和风元素", "日系街头风格", "制服风"],
            "禁忌风格": ["过于暴露", "不尊重传统文化的图案"],
            "颜色偏好": ["红色", "黑色", "白色", "蓝色", "樱花粉"],
            "风格描述": "日本服饰注重细节和层次感，传统和服讲究搭配规则，现代风格多样化。"
        },
        "印度": {
            "推荐风格": ["纱丽", "库尔塔", "莱恩加", "印度传统服饰", "宝莱坞风格"],
            "禁忌风格": ["过于暴露", "牛皮制品", "不尊重宗教信仰的图案"],
            "颜色偏好": ["红色", "金色", "绿色", "黄色", "橙色"],
            "风格描述": "印度服饰色彩鲜艳，图案丰富，注重细节装饰和层次感。"
        },
        "中东": {
            "推荐风格": ["阿巴亚", "长袍", "头巾", "保守现代风格", "传统民族服饰"],
            "禁忌风格": ["暴露", "紧身", "透视", "带有宗教冒犯性的图案"],
            "颜色偏好": ["黑色", "白色", "米色", "深蓝", "深绿"],
            "风格描述": "中东服饰强调保守和端庄，注重遮盖身体，同时保持优雅。"
        },
        "欧洲": {
            "推荐风格": ["欧式休闲", "商务正装", "街头时尚", "高级定制", "复古风格"],
            "禁忌风格": ["带有冒犯性符号的服装", "极端政治符号"],
            "颜色偏好": ["黑色", "白色", "灰色", "蓝色", "米色"],
            "风格描述": "欧洲服饰注重剪裁和质感，强调个人风格表达，既有传统正装也有前卫设计。"
        },
        "北美": {
            "推荐风格": ["休闲风格", "运动风格", "商务休闲", "街头潮流", "波西米亚风"],
            "禁忌风格": ["带有冒犯性文化挪用的服装", "极端政治符号"],
            "颜色偏好": ["多样化", "牛仔蓝", "黑色", "白色", "红色"],
            "风格描述": "北美服饰多元化，强调舒适和个性表达，休闲风格广受欢迎。"
        },
        "非洲": {
            "推荐风格": ["非洲传统服饰", "安卡拉面料服装", "部落风格元素", "现代非洲融合风格"],
            "禁忌风格": ["不尊重部落文化的图案", "过度文化挪用"],
            "颜色偏好": ["鲜艳色彩", "红色", "绿色", "黄色", "橙色"],
            "风格描述": "非洲服饰色彩丰富，图案独特，注重文化象征意义和传统工艺。"
        },
        "东南亚": {
            "推荐风格": ["传统民族服饰", "热带风格", "宽松舒适", "自然材质", "民族元素融合"],
            "禁忌风格": ["过于暴露", "带有宗教冒犯性的图案"],
            "颜色偏好": ["明亮色彩", "自然色调", "热带花卉图案"],
            "风格描述": "东南亚服饰注重舒适和实用，融合传统元素与现代设计，适应热带气候。"
        },
        # 扩展地区数据库 - 细分区域
        "韩国": {
            "推荐风格": ["韩服", "现代韩式时尚", "韩流街头风格", "简约优雅", "层次搭配"],
            "禁忌风格": ["过于暴露", "不尊重传统文化的图案"],
            "颜色偏好": ["白色", "淡雅色调", "粉色", "蓝色", "黑色"],
            "风格描述": "韩国服饰融合传统与现代，注重层次感和简约美学，街头时尚影响全球。"
        },
        "俄罗斯": {
            "推荐风格": ["传统俄罗斯服饰", "萨拉凡", "现代俄式元素", "保暖实用", "华丽刺绣"],
            "禁忌风格": ["过于暴露", "带有政治敏感符号"],
            "颜色偏好": ["红色", "金色", "蓝色", "白色", "深色调"],
            "风格描述": "俄罗斯服饰注重保暖功能和华丽装饰，传统服饰色彩鲜艳，现代风格融合欧洲元素。"
        },
        "拉丁美洲": {
            "推荐风格": ["传统民族服饰", "热带风情", "鲜艳色彩", "流苏装饰", "刺绣元素"],
            "禁忌风格": ["文化挪用", "带有冒犯性的图案"],
            "颜色偏好": ["鲜艳色彩", "红色", "黄色", "绿色", "蓝色"],
            "风格描述": "拉丁美洲服饰色彩鲜艳，充满活力，融合当地文化元素和现代设计。"
        },
        "北欧": {
            "推荐风格": ["简约设计", "功能主义", "自然色调", "高质感材质", "层次搭配"],
            "禁忌风格": ["过度装饰", "不环保材质"],
            "颜色偏好": ["白色", "灰色", "黑色", "蓝色", "自然色调"],
            "风格描述": "北欧服饰注重简约、功能性和可持续性，设计简洁大方，色彩以中性色调为主。"
        }
    }
    
    # 风格转换提示词映射 - 增强版
    STYLE_PROMPT_MAPPING = {
        "中国": "traditional Chinese clothing style, hanfu, qipao, elegant, modest, Chinese cultural elements, silk fabric, embroidery details, mandarin collar, frog buttons, traditional patterns",
        "日本": "Japanese clothing style, kimono elements, modest, elegant layers, yukata, obi belt, Japanese patterns, clean lines, minimalist aesthetic, traditional Japanese fabric",
        "印度": "Indian clothing style, saree, kurta, colorful, traditional patterns, embroidery, gold accents, draping fabric, rich textures, cultural motifs, ornate details",
        "中东": "Middle Eastern modest fashion, covered, elegant, traditional, flowing fabrics, embroidered details, modest cuts, cultural patterns, layered clothing, rich textures",
        "欧洲": "European fashion, elegant, tailored, classic, refined silhouettes, quality fabrics, structured design, minimalist aesthetic, sophisticated details, timeless style",
        "北美": "North American casual style, comfortable, practical, individual, denim, layered looks, relaxed fit, urban elements, versatile pieces, contemporary design",
        "非洲": "African clothing style, ankara fabric, vibrant colors, traditional patterns, bold prints, cultural motifs, wrapped garments, statement accessories, handcrafted details",
        "东南亚": "Southeast Asian style, comfortable, natural materials, traditional elements, batik patterns, loose-fitting, tropical aesthetic, handwoven fabrics, cultural motifs",
        "韩国": "Korean fashion, hanbok elements, modern K-fashion, layered styling, minimalist aesthetic, clean silhouettes, soft colors, street style influence, contemporary design",
        "俄罗斯": "Russian traditional elements, sarafan, ornate embroidery, practical design, layered clothing, rich textures, cultural patterns, warm fabrics, decorative details",
        "拉丁美洲": "Latin American style, vibrant colors, tropical elements, flowing fabrics, embroidered details, cultural motifs, relaxed silhouettes, handcrafted accents",
        "北欧": "Nordic minimalist design, functional, clean lines, sustainable materials, neutral colors, practical elements, quality construction, timeless aesthetic"
    }
    
    # 禁忌风格转换为负面提示词 - 增强版
    TABOO_NEGATIVE_PROMPTS = {
        "中国": "revealing clothes, culturally insensitive patterns, inappropriate symbols, disrespectful imagery, offensive cultural references, historically inaccurate elements",
        "日本": "overly revealing, culturally insensitive, inappropriate representation of traditional elements, offensive imagery, disrespectful use of cultural symbols",
        "印度": "revealing clothes, cow leather, religious insensitive patterns, beef, offensive to Hindu culture, disrespectful imagery, inappropriate use of religious symbols",
        "中东": "revealing clothes, tight fitting, transparent fabrics, religious offensive symbols, immodest attire, culturally inappropriate elements, disrespectful imagery",
        "欧洲": "offensive symbols, extreme political symbols, culturally insensitive representations, inappropriate historical references",
        "北美": "cultural appropriation, offensive symbols, extreme political symbols, insensitive cultural references, stereotypical representations",
        "非洲": "disrespectful tribal patterns, excessive cultural appropriation, stereotypical imagery, colonial references, inappropriate use of cultural symbols",
        "东南亚": "revealing clothes, religious offensive symbols, culturally insensitive patterns, inappropriate use of traditional elements",
        "韩国": "culturally insensitive patterns, inappropriate use of traditional elements, disrespectful imagery, historically inaccurate representations",
        "俄罗斯": "politically sensitive symbols, culturally insensitive representations, inappropriate historical references, offensive imagery",
        "拉丁美洲": "cultural stereotypes, inappropriate cultural appropriation, offensive imagery, disrespectful use of cultural symbols",
        "北欧": "excessive ornamentation, environmentally unsustainable elements, culturally insensitive representations, inappropriate use of traditional symbols"
    }

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点输入类型
        """
        regions = list(cls.STYLE_DATABASE.keys())
        regions.sort()  # 按字母顺序排序地区列表
        
        return {
            "required": {
                "image": ("IMAGE",),
                "region": (regions, {"default": "中国"}),
                "adaptation_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_image_processing": (["是", "否"], {"default": "是"}),
            },
            "optional": {
                "prompt": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "ddim"], {"default": "euler_ancestral"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("adapted_image", "adapted_prompt", "adapted_negative_prompt", "seed")
    FUNCTION = "adapt_clothing_style"
    CATEGORY = "image/style"

    def adapt_clothing_style(self, image, region, adaptation_strength=0.75, enable_image_processing="是", 
                           prompt="", negative_prompt="", seed=0, steps=20, cfg=7.0, sampler_name="euler_ancestral"):
        """
        根据指定地区调整服装风格
        
        参数:
            image: 输入图像
            region: 目标地区
            adaptation_strength: 调整强度
            enable_image_processing: 是否启用图像处理
            prompt: 原始提示词
            negative_prompt: 原始负面提示词
            seed: 随机种子
            steps: 采样步数
            cfg: CFG比例
            sampler_name: 采样器名称
            
        返回:
            调整后的图像、调整后的提示词、调整后的负面提示词、使用的种子
        """
        # 获取地区风格信息
        region_style = self.STYLE_DATABASE.get(region, self.STYLE_DATABASE["中国"])
        
        # 生成随机种子（如果未提供）
        if seed == 0:
            seed = torch.randint(0, 0xffffffffffffffff, (1,)).item()
        
        # 调整提示词，融入地区风格
        style_prompt = self.STYLE_PROMPT_MAPPING.get(region, "")
        taboo_negative = self.TABOO_NEGATIVE_PROMPTS.get(region, "")
        
        # 根据调整强度混合原始提示词和风格提示词
        if prompt:
            # 提取原始提示词的主要内容
            main_subject = self._extract_main_subject(prompt)
            # 根据调整强度混合
            if adaptation_strength > 0.8:
                # 高强度：风格为主
                adapted_prompt = f"{main_subject}, {style_prompt}, {prompt}"
            elif adaptation_strength > 0.5:
                # 中强度：平衡混合
                adapted_prompt = f"{prompt}, {style_prompt}"
            else:
                # 低强度：原始提示词为主，轻微添加风格
                adapted_prompt = f"{prompt}, slight {style_prompt}"
        else:
            adapted_prompt = style_prompt
            
        # 如果有原始负面提示词，与禁忌风格融合
        if negative_prompt:
            adapted_negative_prompt = f"{negative_prompt}, {taboo_negative}"
        else:
            adapted_negative_prompt = taboo_negative
        
        # 图像处理逻辑
        adapted_image = image
        if enable_image_processing == "是":
            # 尝试调用ComfyUI的图像处理节点进行风格转换
            try:
                # 这里我们使用简单的图像处理来模拟风格转换
                # 在实际应用中，这里应该调用适当的图像生成或修改模型
                adapted_image = self._apply_style_filter(image, region, adaptation_strength)
            except Exception as e:
                print(f"图像处理过程中出错: {str(e)}")
                # 如果处理失败，返回原始图像
                adapted_image = image
        
        # 输出风格调整信息
        print(f"调整区域: {region}")
        print(f"推荐风格: {', '.join(region_style['推荐风格'])}")
        print(f"禁忌风格: {', '.join(region_style['禁忌风格'])}")
        print(f"风格描述: {region_style['风格描述']}")
        print(f"调整强度: {adaptation_strength}")
        print(f"调整后提示词: {adapted_prompt}")
        print(f"调整后负面提示词: {adapted_negative_prompt}")
        print(f"使用种子: {seed}")
        
        # 保存风格预览图像（如果需要）
        self._save_style_preview(adapted_image, region, seed)
        
        return (adapted_image, adapted_prompt, adapted_negative_prompt, seed)

    def _extract_main_subject(self, prompt):
        """
        从提示词中提取主要主题
        
        参数:
            prompt: 原始提示词
            
        返回:
            提取的主要主题
        """
        # 简单实现：提取前几个词作为主要主题
        words = prompt.split(',')
        if len(words) > 0:
            return words[0].strip()
        return prompt

    def _apply_style_filter(self, image, region, strength):
        """
        应用风格滤镜到图像
        
        参数:
            image: 输入图像张量
            region: 目标地区
            strength: 调整强度
            
        返回:
            处理后的图像张量
        """
        # 获取地区颜色偏好
        region_style = self.STYLE_DATABASE.get(region, self.STYLE_DATABASE["中国"])
        color_preferences = region_style.get("颜色偏好", [])
        
        # 转换为NumPy数组进行处理
        # 假设image是形状为[batch, height, width, 3]的张量
        img_np = image.cpu().numpy()
        
        # 根据地区特点应用简单的颜色调整
        # 这只是一个简化的示例，实际应用中应该使用更复杂的图像处理或生成模型
        if "红色" in color_preferences:
            # 增强红色通道
            img_np[:, :, :, 0] = np.clip(img_np[:, :, :, 0] * (1 + 0.2 * strength), 0, 1)
        if "蓝色" in color_preferences:
            # 增强蓝色通道
            img_np[:, :, :, 2] = np.clip(img_np[:, :, :, 2] * (1 + 0.2 * strength), 0, 1)
        if "绿色" in color_preferences:
            # 增强绿色通道
            img_np[:, :, :, 1] = np.clip(img_np[:, :, :, 1] * (1 + 0.2 * strength), 0, 1)
        
        # 转回PyTorch张量
        return torch.from_numpy(img_np).to(image.device)

    def _get_output_directory(self):
        """
        获取输出目录路径
        
        返回:
            输出目录的绝对路径
        """
        # 默认输出目录设置在当前工作目录下的'outputs'文件夹
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "outputs")
        return output_dir
        
    def _save_style_preview(self, image, region, seed):
        """
        保存风格预览图像
        
        参数:
            image: 处理后的图像
            region: 目标地区
            seed: 使用的种子
        """
        try:
            # 创建预览目录（如果不存在）
            output_dir = self._get_output_directory()
            preview_dir = os.path.join(output_dir, "style_previews")
            os.makedirs(preview_dir, exist_ok=True)
            
            # 保存预览图像
            # 假设image是形状为[1, height, width, 3]的张量
            img_np = image[0].cpu().numpy()
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # 生成文件名
            filename = f"style_preview_{region}_{seed}.png"
            filepath = os.path.join(preview_dir, filename)
            
            # 保存图像
            img_pil.save(filepath)
            print(f"风格预览已保存至: {filepath}")
        except Exception as e:
            print(f"保存风格预览时出错: {str(e)}")

    def get_style_info(self, region):
        """
        获取指定地区的风格信息
        
        参数:
            region: 地区名称
            
        返回:
            风格信息字典
        """
        return self.STYLE_DATABASE.get(region, {})

    def get_all_regions(self):
        """
        获取所有支持的地区列表
        
        返回:
            地区名称列表
        """
        return list(self.STYLE_DATABASE.keys())

# 节点列表，用于注册到ComfyUI
NODE_CLASS_MAPPINGS = {
    "ClothingStyleAdapter": ClothingStyleAdapter
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothingStyleAdapter": "服装风格自适应器"
}