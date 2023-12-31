# Dream_of_the_Red_Chamber_Character_Analysis
本项目旨在对《红楼梦》这部中国古典小说进行人物关系的可视化分析。通过对小说文本的数据采集、清洗和转换，以及分词处理和词频统计等步骤，最终生成人物关系图，并进行可视化展示。另外，项目还尝试通过数据分析的方法并且可视化来验证《红楼梦》后四十回的作者是否为曹雪芹。
## 文件使用方法
文件最主要的只有两个python文件，DataCollection.py和data_clean_transformation.py。  
其中DataCollection.py用来爬取红楼梦数据，处理爬取的数据，创建“红楼梦”文件夹，将每一章分别保存为txt文件。
data_clean_transformation.py用来处理爬取的数据并且可视化展示。由于使用了Plotly notebook 模式，需要安装Jupyter Notebook  
pip install jupyter  
并且在终端启动Jupyter Notebook  
jupyter notebook  
使用vscode  
版本python3.10.11
### 1. 创建python虚拟环境 
virtualenv -p python venv  
激活venv虚拟环境  
venv\Scripts\activate  
### 项目依赖

下面是项目所需的依赖包列表：（也许会有多余的依赖，建议自己根据报错自己pip）
- anyio==4.0.0
- argon2-cffi==23.1.0
- argon2-cffi-bindings==21.2.0
- arrow==1.2.3
- asttokens==2.4.0
- async-lru==2.0.4
- attrs==23.1.0
- Babel==2.12.1
- backcall==0.2.0
- beautifulsoup4==4.12.2
- bleach==6.0.0
- bs4==0.0.1
- certifi==2023.7.22
- cffi==1.15.1
- charset-normalizer==3.2.0
- colorama==0.4.6
- comm==0.1.4
- contourpy==1.1.0
- cycler==0.11.0
- debugpy==1.7.0
- decorator==5.1.1
- defusedxml==0.7.1
- exceptiongroup==1.1.3
- executing==1.2.0
- fastjsonschema==2.18.0
- fonttools==4.42.1
- fqdn==1.5.1
- idna==3.4
- ipykernel==6.25.2
- ipython==8.15.0
- ipython-genutils==0.2.0
- ipywidgets==8.1.0
- isoduration==20.11.0
- jedi==0.19.0
- jieba==0.42.1
- Jinja2==3.1.2
- joblib==1.3.2
- json5==0.9.14
- jsonpointer==2.4
- jsonschema==4.19.0
- jsonschema-specifications==2023.7.1
- jupyter==1.0.0
- jupyter-console==6.6.3
- jupyter-events==0.7.0
- jupyter-lsp==2.2.0
- jupyter_client==8.3.1
- jupyter_core==5.3.1
- jupyter_server==2.7.3
- jupyter_server_terminals==0.4.4
- jupyterlab==4.0.5
- jupyterlab-pygments==0.2.2
- jupyterlab-widgets==3.0.8
- jupyterlab_server==2.24.0
- kiwisolver==1.4.5
- lxml==4.9.3
- MarkupSafe==2.1.3
- matplotlib==3.7.2
- matplotlib-inline==0.1.6
- mistune==3.0.1
- nbclient==0.8.0
- nbconvert==7.8.0
- nbformat==5.9.2
- nest-asyncio==1.5.7
- networkx==3.1
- notebook==7.0.3
- notebook_shim==0.2.3
- numpy==1.25.2
- overrides==7.4.0
- packaging==23.1
- pandocfilters==1.5.0
- parso==0.8.3
- pickleshare==0.7.5
- Pillow==10.0.0
- platformdirs==3.10.0
- plotly==5.16.1
- prometheus-client==0.17.1
- prompt-toolkit==3.0.39
- psutil==5.9.5
- pure-eval==0.2.2
- pycparser==2.21
- Pygments==2.16.1
- pyparsing==3.0.9
- python-dateutil==2.8.2
- python-json-logger==2.0.7
- pywin32==306
- pywinpty==2.0.11
- PyYAML==6.0.1
- pyzmq==25.1.1
- qtconsole==5.4.4
- QtPy==2.4.0
- referencing==0.30.2
- requests==2.31.0
- rfc3339-validator==0.1.4
- rfc3986-validator==0.1.1
- rpds-py==0.10.2
- scikit-learn==1.3.0
- scipy==1.11.2
- Send2Trash==1.8.2
- six==1.16.0
- sniffio==1.3.0
- soupsieve==2.5
- stack-data==0.6.2
- tenacity==8.2.3
- terminado==0.17.1
- threadpoolctl==3.2.0
- tinycss2==1.2.1
- tomli==2.0.1
- tornado==6.3.3
- tqdm==4.66.1
- traitlets==5.9.0
- typing_extensions==4.7.1
- uri-template==1.3.0
- urllib3==2.0.4
- wcwidth==0.2.6
- webcolors==1.13
- webencodings==0.5.1
- websocket-client==1.6.2
- widgetsnbextension==4.0.8


### 2.运行数据收集文件
python .\DataCollection.py  
### 3.运行数据处理并可视化文件
需要启动Jupyter Notebook  
终端输入命令：	jupyter notebook  
 将会跳转到网页  
 在VS Code中运行代码时，由于Plotly绘图库需要在Notebook环境中才能显示图像，所以无法直接在VS Code中显示图形。可以将该代码片段复制到一个Jupyter Notebook中，并在Jupyter Notebook中执行代码，以便正确显示网络图。  

确保已经安装了必要的依赖库（如plotly、networkx等），然后按照以下步骤操作：

1. 打开Jupyter Notebook。
2. 在Jupyter Notebook中创建一个新的Notebook文件。
3. 将data_clean_transformation.py中代码粘贴到新的Notebook单元格中。
4. shift+enter执行单元格中的代码。
5. 等待显示网络图。

执行代码后，您应该能够在Jupyter Notebook中看到生成的网络图。注意，确保已经正确安装了Plotly和相关依赖库，并且安装的版本与代码兼容。
