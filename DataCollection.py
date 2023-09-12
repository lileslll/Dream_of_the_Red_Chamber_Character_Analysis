import requests
import re
import os
from bs4 import BeautifulSoup
from lxml import etree
#notebook 模块是 tqdm 库中的一个子模块，专门用于在 Jupyter Notebook 或 JupyterLab 环境中显示进度条。
from tqdm import tqdm

# 运行创建目录存储数据
if not os.path.exists('红楼梦'):
    os.mkdir('红楼梦')
    
url = 'http://www.gudianmingzhu.com/guji/hongloumeng/'
# 2.UA伪装
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36'
}
# 3.发送请求&获取响应数据
page_text = requests.get(url=url, headers=headers)
page_text.encoding = page_text.apparent_encoding # 获取编码
page_text = page_text.text
# 4.对text页面进行章节标题文本提取并获取每个章节对应的url链接
soup = BeautifulSoup(page_text, 'lxml')
aTagList = soup.select('#leftdg > div.dgcon.ui > div > a') # 获取a标签信息
titleList = [i.text for i in aTagList] # 获取a标签中的文本信息
urlList = [i["href"] for i in aTagList] # 获取s标签中每个章节的url



# 初始化进度条
progress_bar = tqdm(total=120)

# 遍历 120 个章节
for index in range(1, 121):
     # 更新进度条
    progress_bar.update(1)
    # 获取标题和链接
    title = titleList[index-1]
    url = urlList[index-1]
    # 发送请求&获取响应信息
    page_text = requests.get(url=url, headers=headers, timeout=10)
    page_text.encoding = page_text.apparent_encoding  # 获取编码
    page_text = page_text.text
    # 构建soup对象进行解析文本
    soup = BeautifulSoup(page_text,'lxml')
    # 找到 的 适用类型的路径
    tmp = soup.select('#leftdg > div:nth-child(1) > div > div')
    content = ''.join([p.text for result in tmp for p in result.find_all('p')])
    titleText = soup.select('#leftdg > div:nth-child(1) > div > div > p:nth-child(1)')[0].text
    
    # 使用正则表达式匹配连续的两个 \u3000
    pattern = re.compile(r'\u3000{2}')

    # 将连续的两个 \u3000 替换为一个换行符
    content = re.sub(pattern, '\n', content)
    
    # 删除标签 \u3000 ,超出了 unicode 编码范围,表示全角空白符
    titleText = titleText.replace('\u3000', '')
    
    #标题
    title = titleList[index-1] + titleText
    


    # 保存到本地
    chapter_path = '红楼梦/{}_{}.txt'.format(index, title)
    # 如果已经爬取了则跳过
    if os.path.exists(chapter_path):
        #continue
        os.remove(chapter_path)
    with open(chapter_path, 'w', encoding='utf-8') as f:
        f.write(content)
        

# 完成进度条
progress_bar.close()