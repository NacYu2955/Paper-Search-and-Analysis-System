#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
腾讯云COS配置设置脚本
"""

import os
import sys
import getpass

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.py')
ENV_PATH = os.path.join(PROJECT_ROOT, '.env')
REQUIREMENTS_PATH = os.path.join(PROJECT_ROOT, 'requirements.txt')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("          腾讯云COS配置设置工具")
    print("=" * 60)
    print()

def get_user_input(prompt, default=""):
    """获取用户输入"""
    if default:
        user_input = input(f"{prompt} (默认: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def create_env_file():
    """创建环境变量文件"""
    print("请提供腾讯云COS的配置信息：")
    print()
    
    # 获取配置信息
    secret_id = get_user_input("请输入SecretId")
    if not secret_id:
        print("❌ SecretId不能为空")
        return False
    
    secret_key = getpass.getpass("请输入SecretKey (输入时不会显示): ")
    if not secret_key:
        print("❌ SecretKey不能为空")
        return False
    
    region = get_user_input("请输入存储桶地域", "ap-guangzhou")
    bucket_name = get_user_input("请输入存储桶名称")
    if not bucket_name:
        print("❌ 存储桶名称不能为空")
        return False
    
    use_cos = get_user_input("是否启用COS存储 (y/n)", "y").lower()
    use_cos_storage = "true" if use_cos in ['y', 'yes', '是'] else "false"
    
    # 创建.env文件内容
    env_content = f"""# 腾讯云COS配置
COS_SECRET_ID={secret_id}
COS_SECRET_KEY={secret_key}
COS_REGION={region}
COS_BUCKET_NAME={bucket_name}

# 存储模式配置
USE_COS_STORAGE={use_cos_storage}
"""
    
    # 写入.env文件
    try:
        with open(ENV_PATH, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ 环境变量文件 .env 创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建环境变量文件失败: {str(e)}")
        return False

def update_config_file():
    """更新config.py文件"""
    print("\n正在更新config.py文件...")
    
    try:
        # 读取当前config.py内容
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经包含COS配置
        if 'COS_SECRET_ID' in content:
            print("✅ config.py文件已包含COS配置")
            return True
        
        # 在文件末尾添加COS配置
        cos_config = '''
# 腾讯云COS配置
COS_SECRET_ID = os.getenv('COS_SECRET_ID', 'your_secret_id_here')
COS_SECRET_KEY = os.getenv('COS_SECRET_KEY', 'your_secret_key_here')
COS_REGION = os.getenv('COS_REGION', 'ap-shanghai')  # 上海地域
COS_BUCKET_NAME = os.getenv('COS_BUCKET_NAME', 'your_bucket_name_here')  # 您的存储桶名称
COS_FOLDER = 'pdfs'  # COS中的文件夹名称

# 是否启用腾讯云COS存储（如果为False，则使用本地存储）
USE_COS_STORAGE = os.getenv('USE_COS_STORAGE', 'true').lower() == 'true'
'''
        
        # 在服务器配置之前插入COS配置
        if '# 服务器配置' in content:
            content = content.replace('# 服务器配置', cos_config + '\n# 服务器配置')
        else:
            content += cos_config
        
        # 写入更新后的内容
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ config.py文件更新成功")
        return True
        
    except Exception as e:
        print(f"❌ 更新config.py文件失败: {str(e)}")
        return False

def install_dependencies():
    """安装依赖包"""
    print("\n正在安装依赖包...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS_PATH],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖包安装成功")
            return True
        else:
            print(f"❌ 依赖包安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 安装依赖包时出错: {str(e)}")
        return False

def test_configuration():
    """测试配置"""
    print("\n正在测试配置...")
    
    try:
        # 加载环境变量
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
        
        # 导入并测试COS功能
        from coding.cos_utils import COSUtils
        from config.config import COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET_NAME
        
        if COS_SECRET_ID == 'your_secret_id_here':
            print("❌ 请先配置SecretId和SecretKey")
            return False
        
        cos_client = COSUtils(COS_SECRET_ID, COS_SECRET_KEY, COS_REGION, COS_BUCKET_NAME)
        print("✅ COS客户端初始化成功")
        
        # 测试连接
        test_content = b"test"
        result = cos_client.upload_file(test_content, "test.txt", "test")
        
        if result['success']:
            print("✅ COS连接测试成功")
            # 清理测试文件
            cos_client.delete_file(result['cos_key'])
            return True
        else:
            print(f"❌ COS连接测试失败: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    print_banner()
    
    print("此工具将帮助您配置腾讯云COS存储功能。")
    print("请确保您已经：")
    print("1. 在腾讯云控制台创建了对象存储COS存储桶")
    print("2. 获取了API密钥（SecretId和SecretKey）")
    print("3. 记录了存储桶名称和地域信息")
    print()
    
    choice = input("是否继续配置？(y/n): ").lower()
    if choice not in ['y', 'yes', '是']:
        print("配置已取消")
        return
    
    print()
    
    # 步骤1: 创建环境变量文件
    print("步骤1: 创建环境变量文件")
    if not create_env_file():
        return
    
    # 步骤2: 更新config.py
    print("\n步骤2: 更新配置文件")
    if not update_config_file():
        return
    
    # 步骤3: 安装依赖
    print("\n步骤3: 安装依赖包")
    if not install_dependencies():
        print("⚠️  依赖包安装失败，请手动运行: pip install -r requirements.txt")
    
    # 步骤4: 测试配置
    print("\n步骤4: 测试配置")
    if test_configuration():
        print("\n🎉 配置完成！")
        print("\n使用说明：")
        print("1. 启动应用: python app.py")
        print("2. 访问: http://localhost:6006")
        print("3. 上传PDF文件将自动存储到腾讯云COS")
        print("\n如需禁用COS存储，请设置环境变量: USE_COS_STORAGE=false")
    else:
        print("\n❌ 配置测试失败，请检查配置信息")
        print("常见问题：")
        print("1. SecretId和SecretKey是否正确")
        print("2. 存储桶是否存在且权限正确")
        print("3. 网络连接是否正常")

if __name__ == "__main__":
    main() 
