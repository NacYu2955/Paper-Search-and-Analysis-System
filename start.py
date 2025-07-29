#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phddns状态检查和启动脚本
功能：
1. 检查phddns状态并显示SN码
2. 如果phddns未安装或未运行，则安装并启动
3. 运行app.py文件
"""

import os
import sys
import subprocess
import time
import platform

def run_command(command, description="", silent=False):
    """执行命令并处理错误"""
    if not silent:
        print(f"正在执行: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        if not silent:
            print(f"✓ {description or command} 执行成功")
            if result.stdout:
                print(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"✗ {description or command} 执行失败")
            print(f"错误信息: {e.stderr}")
        return False

def check_file_exists(filename):
    """检查文件是否存在"""
    if not os.path.exists(filename):
        print(f"✗ 文件 {filename} 不存在")
        return False
    print(f"✓ 文件 {filename} 存在")
    return True

def check_phddns_status():
    """检查phddns状态并显示SN码"""
    print("正在检查phddns状态...")
    
    # 检查phddns是否已安装
    if not run_command("which phddns", "检查phddns是否已安装", silent=True):
        print("✗ phddns未安装")
        return False
    
    print("✓ phddns已安装")
    
    # 检查phddns服务状态
    print("正在检查phddns服务状态...")
    status_result = subprocess.run("sudo phddns status", shell=True, 
                                  capture_output=True, text=True, encoding='utf-8')
    
    if status_result.returncode == 0:
        print("✓ phddns服务状态:")
        print(status_result.stdout)
        
        # 尝试获取SN码
        print("正在获取SN码...")
        sn_result = subprocess.run("sudo phddns status | grep -i sn", shell=True, 
                                  capture_output=True, text=True, encoding='utf-8')
        
        if sn_result.returncode == 0 and sn_result.stdout.strip():
            print("✓ SN码信息:")
            print(sn_result.stdout.strip())
        else:
            # 尝试其他方式获取SN码
            print("尝试其他方式获取SN码...")
            alternative_sn_result = subprocess.run("sudo phddns status | grep -E 'SN|sn|Serial'", shell=True, 
                                                  capture_output=True, text=True, encoding='utf-8')
            if alternative_sn_result.returncode == 0 and alternative_sn_result.stdout.strip():
                print("✓ SN码信息:")
                print(alternative_sn_result.stdout.strip())
            else:
                print("⚠ 无法获取SN码信息，但服务正在运行")
    else:
        print("✗ phddns服务未运行或状态检查失败")
        print(f"错误信息: {status_result.stderr}")
        return False
    
    return True

def install_phddns():
    """安装phddns包"""
    deb_file = "phddns_5.3.0_amd64.deb"
    
    # 检查deb文件是否存在
    if not check_file_exists(deb_file):
        print(f"请确保 {deb_file} 文件在当前目录中")
        return False
    
    # 使用sudo dpkg -i安装deb包
    if not run_command(f"sudo dpkg -i {deb_file}", f"安装 {deb_file}"):
        # 如果安装失败，尝试修复依赖
        print("尝试修复依赖关系...")
        if not run_command("sudo apt-get update", "更新包列表"):
            return False
        if not run_command("sudo apt-get install -f -y", "修复依赖关系"):
            return False
        # 重新尝试安装
        if not run_command(f"sudo dpkg -i {deb_file}", f"重新安装 {deb_file}"):
            return False
    
    return True

def start_phddns():
    """启动phddns服务"""
    # 检查phddns是否已安装
    if not run_command("which phddns", "检查phddns是否已安装"):
        print("phddns未找到，请确保安装成功")
        return False
    
    # 启动phddns服务
    if not run_command("sudo phddns start", "启动phddns服务"):
        print("phddns启动失败")
        return False
    
    # 等待服务启动
    time.sleep(2)
    print("phddns服务启动完成")
    
    return True

def run_python_script():
    """运行app.py文件并显示实时输出"""
    python_file = "app.py"
    
    # 检查app.py文件是否存在
    if not check_file_exists(python_file):
        print(f"请确保 {python_file} 文件在当前目录中")
        return False
    
    print(f"\n正在启动 {python_file}...")
    print("=" * 50)
    print("应用启动日志:")
    print("=" * 50)
    
    try:
        # 使用subprocess.Popen来实时显示输出
        process = subprocess.Popen(
            [sys.executable, python_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时读取并显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程结束并获取返回码
        return_code = process.poll()
        
        if return_code == 0:
            print("\n" + "=" * 50)
            print("应用正常退出")
            print("=" * 50)
            return True
        else:
            print(f"\n" + "=" * 50)
            print(f"应用异常退出，返回码: {return_code}")
            print("=" * 50)
            return False
            
    except KeyboardInterrupt:
        print("\n用户中断了应用运行")
        if 'process' in locals():
            process.terminate()
            process.wait()
        return False
    except Exception as e:
        print(f"\n启动应用时发生错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("phddns状态检查和启动脚本")
    print("=" * 50)
    
    # 检查是否可以使用sudo（安装deb包需要sudo权限）
    if not run_command("sudo -n true", "检查sudo权限", silent=True):
        print("警告: 此脚本需要sudo权限来安装deb包和启动服务")
        print("请确保当前用户有sudo权限，或者使用 sudo python3 start.py 运行此脚本")
        return False
    
    # 检查操作系统
    if platform.system() != "Linux":
        print("警告: 此脚本设计用于Linux系统")
        return False
    
    try:
        # 步骤1: 检查phddns状态
        print("\n步骤1: 检查phddns状态")
        if not check_phddns_status():
            print("phddns状态检查失败，尝试安装...")
            # 如果状态检查失败，尝试安装
            print("\n步骤1.1: 安装phddns包")
            if not install_phddns():
                print("安装phddns失败，退出脚本")
                return False
            
            print("\n步骤1.2: 启动phddns服务")
            if not start_phddns():
                print("启动phddns失败，退出脚本")
                return False
            
            # 重新检查状态
            print("\n重新检查phddns状态...")
            if not check_phddns_status():
                print("phddns状态检查仍然失败，退出脚本")
                return False
        
        # 步骤2: 运行app.py
        print("\n步骤2: 运行app.py")
        if not run_python_script():
            print("运行app.py失败")
            return False
        
        print("\n" + "=" * 50)
        print("所有步骤执行完成！")
        print("=" * 50)
        return True
        
    except KeyboardInterrupt:
        print("\n用户中断了脚本执行")
        return False
    except Exception as e:
        print(f"\n脚本执行过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
