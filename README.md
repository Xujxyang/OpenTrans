# OpenTrans
This repo contains the code for our paper [**Transferable and Principled Efficiency for Open-Vocabulary Segmentation**]

<div align="center">
  <img src="imgs/structure9.jpg" width="90%" height="90%"/>
</div><br/>

## Abstract
Recent success of pre-trained foundation vision-language models makes Open-Vocabulary Segmentation (OVS) possible. Despite the promising performance, this approach introduces heavy computational overheads for two challenges: 1) {large model sizes} of the backbone; 2) expensive costs during the {fine-tuning}. These challenges hinder this OVS strategy from being widely applicable and affordable in real-world scenarios. Although traditional methods such as model compression and efficient fine-tuning can address these challenges, they often rely on heuristics. This means that their solutions cannot be easily transferred and necessitate re-training on different models, which comes at a cost. In the context of efficient OVS, we target achieving performance that is comparable to or even better than prior OVS works based on large vision-language foundation models, by utilizing \textbf{smaller models that incur lower training costs}. The core strategy is to make our efficiency \textbf{principled} and thus seamlessly \textbf{transferable} from one OVS framework to others without further customization. Comprehensive experiments on diverse OVS benchmarks demonstrate our superior trade-off between segmentation accuracy and computation costs over previous works.
 
## Installation
You can configure the dataset and runtime environment based on the settings provided in [fc-clip](https://github.com/bytedance/fc-clip). 

## Result
<table>
<thead>
  <tr>
    <td align="center">Model</td>
    <td align="center">COCO</td>
    <td align="center">Cityscapes</td>
    <td align="center">ADE20K-150</td>
    <td align="center">ADE20K-847</td>
    <td align="center">PAS-20</td>
    <td align="center">PC-59</td>
    <td align="center">PC-459</td>
    <td align="center">Params</td>
    <td align="center">FLOPs</td>
  </tr>
</thead>
<tbody>
</tbody>
</table>
