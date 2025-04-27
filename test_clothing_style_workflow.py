#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服装风格工作流节点测试脚本
用于验证服装风格工作流节点的功能是否正常
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# 导入服装风格适配器和工作流类
try:
    from clothing_style_adapter import ClothingStyleAdapter
    from clothing_style_workflow import ClothingStyleWorkflow, StyleInfoDisplay, RegionalStyleSelector
    print("成功导入服装风格工作流模块")
except ImportError as e:
    print(f"导入服装风格工作流模块失败: {str(e)}")
    sys.exit(1)

def test_clothing_style_workflow():
    """
    测试服装风格工作流节点的基本功能
    """
    print("\n===== 开始测试服装风格工作流节点 =====")
    
    # 创建测试图像
    print("创建测试图像...")
    test_image = torch.zeros(1, 512, 512, 3)
    test_image = test_image.float()
    
    # 创建工作流实例
    print("创建工作流实例...")
    workflow = ClothingStyleWorkflow()
    
    # 测试不同地区
    regions = ["中国", "日本", "印度", "欧洲", "北美"]
    
    for region in regions:
        print(f"\n测试 {region} 地区的服装风格调整...")
        
        # 测试完整工作流模式
        print("测试完整工作流模式...")
        result = workflow.run_workflow(
            image=test_image,
            region=region,
            adaptation_strength=0.75,
            enable_image_processing="是",
            workflow_mode="完整工作流",
            prompt="a person wearing clothes",
            negative_prompt="",
            seed=42,
            steps=20,
            cfg=7.0,
            sampler_name="euler_ancestral",
            width=512,
            height=512,
            batch_size=1
        )
        
        # 检查返回值
        adapted_image, adapted_prompt, adapted_negative_prompt, seed, style_info = result
        
        print(f"调整后提示词: {adapted_prompt}")
        print(f"调整后负面提示词: {adapted_negative_prompt}")
        print(f"使用种子: {seed}")
        
        # 测试风格信息显示器
        print("测试风格信息显示器...")
        info_display = StyleInfoDisplay()
        style_text = info_display.display_style_info(style_info, "完整信息")
        print(f"风格信息文本长度: {len(style_text[0])} 字符")
        
        # 测试地区风格选择器
        print("测试地区风格选择器...")
        selector = RegionalStyleSelector()
        style_prompt, negative_prompt, selector_style_info = selector.select_regional_style(region, "全部")
        print(f"选择器风格提示词: {style_prompt[:50]}...")
        print(f"选择器负面提示词: {negative_prompt[:50]}...")
    
    print("\n===== 服装风格工作流节点测试完成 =====")
    print("所有测试通过!")

def main():
    """
    主函数
    """
    print("开始测试服装风格工作流节点系统...")
    test_clothing_style_workflow()
    print("测试完成!")

if __name__ == "__main__":
    main()