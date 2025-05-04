from modelscope.msdatasets import MsDataset
from modelscope.hub.api import HubApi

# 登录
api = HubApi()
api.login('7e37438b-c7b7-4a0e-9276-f719247969b2')  # 备注：my-sdk-token 需要从modelscope-个人中心-访问令牌获取
# thejiangcj/FakeThread
# 删除单个文件（其中my-data.zip为您上传到数据集-数据文件的压缩文件名称）
# MsDataset.delete(object_name='my-data.zip', dataset_name='my-dataset-name', namespace='my-namespace')

# 删除文件夹（其中my-data-dir为您上传到数据集-数据文件的文件夹名称）
# MsDataset.delete(object_name='video', dataset_name='FakeThread', namespace='thejiangcj')

# 删除文件夹中的某个文件
MsDataset.delete(object_name='video/train/Youku-mPLUG_part33.tar.gz', dataset_name='FakeThread', namespace='thejiangcj')