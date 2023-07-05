调用yolov5部分
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

调用deepsort部分
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort







在main函数下面首先是初始化配置代码块
Help都有解释


然后执行
之后就是判断符合条件的就放入对应的列表，注意不符合条件被记录过的要及时删掉
先画出ROI_box的速度，距离
再之后就是总体图像上标注的代码块
方便调参写的

画出红线，和写出数据的方便调参的输出
常用修改的大体框架就这些

框上的数据是
图像上的标注基本都是画框代码块下面的进行描绘。

ROI框的颜色判断和修改，还有框上的警告标注都是
Draw这个函数里执行的，其他的总体图像上的标注，比如

配合红线调参line，需要注意的是距离限制和单目测距的实际距离结合使用判断是不是在危险距离，线是图像坐标判断是不是在盲区里

draw代码里面判断了符合什么条件何时标红。并在框上标出对应警告。
注意：测速测得速度是相对速度，和摄像头的相对速度

