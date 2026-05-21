import logging
import os
import uuid
import requests
from io import BytesIO
from datetime import datetime, timezone
import time
import json
import hashlib
import hmac
import random
from urllib.parse import urlencode
import base64

logger = logging.getLogger(__name__)

class COSUtils:
    def __init__(self, secret_id, secret_key, region, bucket_name):
        """
        初始化COS工具类
        
        Args:
            secret_id: 腾讯云SecretId
            secret_key: 腾讯云SecretKey
            region: COS地域
            bucket_name: 存储桶名称
        """
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.bucket_name = bucket_name
        
        # 初始化COS客户端
        try:
            from qcloud_cos import CosConfig, CosS3Client
            config = CosConfig(
                Region=region,
                SecretId=secret_id,
                SecretKey=secret_key
            )
            self.client = CosS3Client(config)
            logger.info(f"COS客户端初始化成功: {bucket_name}")
        except Exception as e:
            logger.error(f"COS客户端初始化失败: {str(e)}")
            self.client = None
    
    def upload_file(self, file_data, file_name, folder="pdfs"):
        """
        上传文件到COS
        
        Args:
            file_data: 文件数据（bytes或文件对象）
            file_name: 文件名
            folder: 存储文件夹，默认为pdfs
            
        Returns:
            dict: 上传结果
        """
        try:
            if not self.client:
                return {'success': False, 'error': 'COS客户端未初始化'}
            
            # 生成COS键名
            cos_key = f"{folder}/{file_name}"
            
            # 检查文件数据
            if isinstance(file_data, bytes):
                logger.info(f"上传文件: {cos_key}, 大小: {len(file_data)} bytes")
                if len(file_data) > 10:
                    logger.info(f"文件前10字节: {file_data[:10]}")
                    if file_data[:4] == b'%PDF':
                        logger.info("确认是PDF文件")
                    else:
                        logger.warning("文件可能不是PDF格式")
            else:
                logger.info(f"上传文件对象: {cos_key}")
            
            # 上传文件
            if isinstance(file_data, bytes):
                response = self.client.put_object(
                    Bucket=self.bucket_name,
                    Body=file_data,
                    Key=cos_key
                )
            else:
                response = self.client.put_object(
                    Bucket=self.bucket_name,
                    Body=file_data,
                    Key=cos_key
                )
            
            logger.info(f"文件上传成功: {cos_key}")
            logger.info(f"上传响应: {response}")
            return {
                'success': True,
                'cos_key': cos_key,
                'etag': response.get('ETag', ''),
                'url': f"https://{self.bucket_name}.cos.{self.region}.myqcloud.com/{cos_key}"
            }
            
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            logger.error(f"异常详情: ", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def download_file(self, cos_key):
        """
        从COS下载文件
        
        Args:
            cos_key: COS中的文件键
            
        Returns:
            bytes: 文件数据
        """
        try:
            if not self.client:
                logger.error("COS客户端未初始化")
                return None
            
            logger.info(f"开始下载文件: {cos_key}, bucket: {self.bucket_name}")
            
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=cos_key
            )
            
            logger.info(f"获取对象响应成功: {cos_key}")
            logger.info(f"响应头: {response}")
            
            # 检查Content-Length
            if 'Content-Length' in response:
                expected_size = int(response['Content-Length'])
                logger.info(f"期望的文件大小: {expected_size} bytes")
            
            file_data = response['Body'].read()
            logger.info(f"文件下载成功: {cos_key}, 大小: {len(file_data)} bytes")
            
            # 检查文件内容
            if len(file_data) > 10:
                logger.info(f"文件前10字节: {file_data[:10]}")
                if file_data[:4] == b'%PDF':
                    logger.info("确认是PDF文件")
                else:
                    logger.warning("文件可能不是PDF格式")
            
            return file_data
            
        except Exception as e:
            logger.error(f"文件下载失败: {str(e)}")
            logger.error(f"异常详情: ", exc_info=True)
            return None
    
    def delete_file(self, cos_key):
        """
        删除COS中的文件
        
        Args:
            cos_key: COS中的文件键
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if not self.client:
                return False
            
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=cos_key
            )
            
            logger.info(f"文件删除成功: {cos_key}")
            return True
            
        except Exception as e:
            logger.error(f"文件删除失败: {str(e)}")
            return False
    
    def get_file_url(self, cos_key, expires=3600, inline=False):
        """
        获取文件的预签名URL
        
        Args:
            cos_key: COS中的文件键
            expires: URL有效期（秒），默认1小时
            inline: 是否在线预览模式，默认False
            
        Returns:
            str: 预签名URL
        """
        try:
            # 设置响应头参数
            params = {}
            if inline:
                # 添加在线预览参数
                params['response-content-disposition'] = 'inline'
                params['response-content-type'] = 'application/pdf'
            
            url = self.client.get_presigned_url(
                Method='GET',
                Bucket=self.bucket_name,
                Key=cos_key,
                Expired=expires,
                Params=params
            )
            
            logger.info(f"生成预签名URL成功: {cos_key}, inline={inline}")
            return url
            
        except Exception as e:
            logger.error(f"生成预签名URL失败: {str(e)}")
            return None
    
    def file_exists(self, cos_key):
        """
        检查文件是否存在
        
        Args:
            cos_key: COS中的文件键
            
        Returns:
            bool: 文件是否存在
        """
        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=cos_key
            )
            return True
        except Exception:
            return False
    
    def get_file_info(self, cos_key):
        """
        获取文件信息
        
        Args:
            cos_key: COS中的文件键
            
        Returns:
            dict: 文件信息
        """
        try:
            response = self.client.head_object(
                Bucket=self.bucket_name,
                Key=cos_key
            )
            
            return {
                'size': response.get('Content-Length', 0),
                'etag': response.get('ETag', ''),
                'last_modified': response.get('Last-Modified', ''),
                'content_type': response.get('Content-Type', '')
            }
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {str(e)}")
            return None

    def get_upload_sts_token(self, allow_prefix='*', duration_seconds=1800):
        """
        使用腾讯云官方SDK生成前端直传COS的临时密钥。
        Args:
            allow_prefix: 允许上传的文件前缀
            duration_seconds: 临时密钥有效期（秒）
        Returns:
            dict: 包含临时密钥和相关信息
        """
        try:
            from python.sts.sts import Sts
            
            config = {
                # 请求URL
                'url': 'https://sts.tencentcloudapi.com/',
                # 域名
                'domain': 'sts.tencentcloudapi.com',
                # 临时密钥有效时长，单位是秒
                'duration_seconds': duration_seconds,
                'secret_id': self.secret_id,
                # 固定密钥
                'secret_key': self.secret_key,
                # 换成你的 bucket
                'bucket': self.bucket_name,
                # 换成 bucket 所在地区
                'region': self.region,
                # 这里改成允许的路径前缀
                'allow_prefix': [allow_prefix],
                # 密钥的权限列表
                'allow_actions': [
                    # 简单上传
                    'name/cos:PutObject',
                    'name/cos:PostObject',
                    # 分片上传
                    'name/cos:InitiateMultipartUpload',
                    'name/cos:ListMultipartUploads',
                    'name/cos:ListParts',
                    'name/cos:UploadPart',
                    'name/cos:CompleteMultipartUpload',
                    'name/cos:AbortMultipartUpload'
                ]
            }
            
            sts = Sts(config)
            response = sts.get_credential()
            response_dict = dict(response)
            
            if 'credentials' in response_dict:
                credential_info = response_dict.get("credentials")
                return {
                    'credentials': {
                        'tmpSecretId': credential_info.get("tmpSecretId"),
                        'tmpSecretKey': credential_info.get("tmpSecretKey"),
                        'sessionToken': credential_info.get("sessionToken"),
                    },
                    'startTime': response_dict.get("startTime"),
                    'expiredTime': response_dict.get("expiredTime"),
                    'requestId': response_dict.get("requestId"),
                    'expiration': response_dict.get("expiration"),
                    'bucket': self.bucket_name,
                    'region': self.region
                }
            else:
                logger.error(f"STS SDK返回格式错误: {response_dict}")
                return None
                
        except Exception as e:
            logger.error(f"生成COS临时密钥失败: {str(e)}")
            return None

    def get_upload_sts_token_via_api(self, allow_prefix='*', duration_seconds=1800):
        """
        通过腾讯云STS云API获取前端直传COS的临时密钥。
        Args:
            allow_prefix: 允许上传的文件前缀
            duration_seconds: 临时密钥有效期（秒）
        Returns:
            dict: 包含临时密钥和相关信息
        """
        import requests
        import time
        import json
        import hashlib
        import hmac
        import random
        from urllib.parse import urlencode
        from datetime import datetime, timezone
        
        # 1. 构造参数
        endpoint = 'https://sts.tencentcloudapi.com/'
        action = 'GetFederationToken'
        version = '2018-08-13'
        region = self.region
        secret_id = self.secret_id
        secret_key = self.secret_key
        name = 'cos-upload-user'
        # 由于服务器时间不准确，直接加10分钟补偿
        timestamp = int(time.time()) + 600
        nonce = random.randint(10000, 99999)
        policy = {
            "version": "2.0",
            "statement": [
                {
                    "effect": "allow",
                    "action": [
                        "name/cos:PutObject",
                        "name/cos:PostObject",
                        "name/cos:InitiateMultipartUpload",
                        "name/cos:ListMultipartUploadParts",
                        "name/cos:UploadPart",
                        "name/cos:CompleteMultipartUpload",
                        "name/cos:AbortMultipartUpload"
                    ],
                    "resource": [
                        f"qcs::cos:{region}:uid/{secret_id}:{self.bucket_name}/*"
                    ]
                }
            ]
        }
        
        # 2. 构建请求体
        request_body = {
            'Action': action,
            'Version': version,
            'Region': region,
            'Timestamp': timestamp,
            'Nonce': nonce,
            'SecretId': secret_id,
            'Name': name,
            'DurationSeconds': duration_seconds,
            'Policy': json.dumps(policy, separators=(',', ':'))
        }
        
        # 3. TC3-HMAC-SHA256签名算法
        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
        
        def get_signature_key(key, date_stamp, region_name, service_name):
            k_date = sign(('TC3' + key).encode('utf-8'), date_stamp)
            k_region = sign(k_date, region_name)
            k_service = sign(k_region, service_name)
            k_signing = sign(k_service, 'tc3_request')
            return k_signing
        
        # 生成签名
        date_stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        credential_scope = f'{date_stamp}/sts/tc3_request'
        
        # 构建规范请求
        http_request_method = 'POST'
        canonical_uri = '/'
        canonical_querystring = ''
        canonical_headers = 'content-type:application/json; charset=utf-8\nhost:sts.tencentcloudapi.com\n'
        signed_headers = 'content-type;host'
        
        payload_hash = hashlib.sha256(json.dumps(request_body).encode('utf-8')).hexdigest()
        canonical_request = f'{http_request_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}'
        
        # 构建待签名字符串
        algorithm = 'TC3-HMAC-SHA256'
        request_timestamp = str(timestamp)
        string_to_sign = f'{algorithm}\n{request_timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
        
        # 计算签名
        signing_key = get_signature_key(self.secret_key, date_stamp, 'sts', 'sts')
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # 构建授权头
        authorization = f'{algorithm} Credential={self.secret_id}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}'
        
        # 4. 发送请求
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Host': 'sts.tencentcloudapi.com',
            'X-TC-Action': action,
            'X-TC-Version': version,
            'X-TC-Timestamp': str(timestamp),
            'Authorization': authorization
        }
        
        logger.info(f"发送STS请求，时间戳: {timestamp}, UTC时间: {datetime.now(timezone.utc)}")
        response = requests.post(endpoint, headers=headers, data=json.dumps(request_body))
        result = response.json()
        
        if 'Response' in result and 'Credentials' in result['Response']:
            cred = result['Response']['Credentials']
            logger.info(f"成功生成临时密钥，有效期: {result['Response'].get('ExpiredTime')}")
            return {
                'credentials': {
                    'tmpSecretId': cred['TmpSecretId'],
                    'tmpSecretKey': cred['TmpSecretKey'],
                    'sessionToken': cred['Token']
                },
                'startTime': result['Response'].get('StartTime'),
                'expiredTime': result['Response'].get('ExpiredTime'),
                'requestId': result['Response'].get('RequestId'),
                'bucket': self.bucket_name,
                'region': self.region
            }
        else:
            logger.error(f"STS API返回错误: {result}")
            return None 