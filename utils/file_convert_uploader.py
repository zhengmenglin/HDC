#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件上传转换脚本
用于查找指定类型文件并上传至网站进行转换，下载转换结果
"""

import os
import glob
import time
import logging
import requests
from urllib.parse import urljoin

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('file_convert.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_files():
    """
    查找脚本所在目录及其子目录下的所有.xxc、.gcb、.cap文件
    :return: 找到的文件路径列表
    """
    logger.info("开始查找目标文件...")
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要查找的文件类型
    file_types = ['*.xxc', '*.gcb', '*.cap']
    
    # 递归查找所有目标文件
    found_files = []
    for file_type in file_types:
        # 使用glob递归查找
        files = glob.glob(os.path.join(script_dir, '**', file_type), recursive=True)
        found_files.extend(files)
    
    logger.info(f"找到 {len(found_files)} 个文件需要转换：")
    for file in found_files:
        logger.info(f"  - {file}")
    
    return found_files


def upload_and_convert(file_path):
    """
    上传文件并进行转换
    :param file_path: 要上传的文件路径
    :return: 转换是否成功
    """
    logger.info(f"开始处理文件：{file_path}")
    
    # 确保下载目录存在
    download_dir = os.path.dirname(file_path)
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logger.info(f"创建下载目录：{download_dir}")
    
    # 目标网站URL
    base_url = "http://member.xtxf.com"
    convert_url = urljoin(base_url, "/index/user/convert.html")
    
    # 创建会话
    session = requests.Session()
    
    try:
        # 1. 获取页面，获取必要的表单数据
        logger.info(f"1. 访问目标网站：{convert_url}")
        response = session.get(convert_url)
        if response.status_code != 200:
            logger.error(f"访问网站失败，状态码：{response.status_code}")
            return False
        
        # 2. 上传文件
        logger.info("2. 上传文件...")
        
        # 构造上传请求
        upload_endpoint = urljoin(base_url, "/index/user/convert.html")
        
        # 准备表单数据
        form_data = {
            "api_id_text": "实训机",
            "submit": "提交"
        }
        
        # 准备文件数据
        files = {
            "filepath": (os.path.basename(file_path), open(file_path, 'rb'), "application/octet-stream")
        }
        
        # 发送上传请求
        logger.info(f"   上传文件：{file_path}")
        response = session.post(upload_endpoint, data=form_data, files=files)
        
        if response.status_code != 200:
            logger.error(f"上传文件失败，状态码：{response.status_code}")
            logger.error(f"响应内容：{response.text[:500]}...")
            return False
        
        logger.info("   文件上传成功")
        
        # 3. 等待转换完成
        logger.info("3. 等待转换完成...")
        time.sleep(10)  # 等待10秒，根据实际情况调整
        
        # 4. 下载转换结果
        logger.info("4. 下载转换结果...")
        
        # 构造下载请求
        # 注意：这里需要根据实际网站的下载链接格式进行调整
        # 假设下载链接格式为 /index/user/download?id=xxx
        # 实际使用时需要分析网站的HTML结构或网络请求来获取正确的下载链接
        
        # 这里我们使用一个简化的方法，直接尝试下载
        # 实际使用时需要根据网站的具体情况进行调整
        download_url = urljoin(base_url, "/index/user/download")
        
        # 发送下载请求
        response = session.get(download_url, stream=True)
        
        if response.status_code == 200:
            # 构造输出文件名
            output_file = os.path.splitext(file_path)[0] + ".json"
            
            # 写入文件
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"   转换结果下载成功，保存至：{output_file}")
            logger.info(f"   文件大小：{os.path.getsize(output_file)} 字节")
            return True
        else:
            logger.error(f"下载转换结果失败，状态码：{response.status_code}")
            logger.error(f"响应内容：{response.text[:500]}...")
            return False
        
    except Exception as e:
        logger.error(f"文件转换失败: {str(e)}", exc_info=True)
        return False
    finally:
        # 关闭文件
        for file_tuple in files.values() if 'files' in locals() else []:
            if hasattr(file_tuple, '__getitem__') and len(file_tuple) > 1:
                file_tuple[1].close()


def main():
    """
    主函数
    """
    logger.info("===== 文件上传转换脚本开始执行 =====")
    
    # 查找目标文件
    files_to_convert = find_files()
    
    if not files_to_convert:
        logger.info("没有找到需要转换的文件")
        return
    
    # 依次上传转换每个文件
    success_count = 0
    total_count = len(files_to_convert)
    
    # 处理所有文件
    for i, file_path in enumerate(files_to_convert, 1):
        logger.info(f"\n正在处理第 {i}/{total_count} 个文件：{os.path.basename(file_path)}")
        
        if upload_and_convert(file_path):
            success_count += 1
        
        # 为避免服务器压力，添加间隔
        if i < total_count:
            logger.info("等待5秒后处理下一个文件...")
            time.sleep(5)
    
    logger.info(f"\n===== 转换完成！成功：{success_count}/{total_count} =====")


if __name__ == "__main__":
    main()