Edible Vs Poisonous and Mushroom vs Non-Mushroom Sporocarp Classification. Download the dataset from https://www.kaggle.com/marcosvolpato/edible-and-poisonous-fungi and extract it into a folder by the name, Data. If extracted into a different folder, specify `--path` during run.

---

&nbsp;

### **CLI Arguments**
<pre>
1. --kaggle      - Flag that allows for usage of scripts within a kaggle notebook on the dataset (Default: None)  
2. --bs          - Batch Size (Default: 64)
3. --lr          - Learing Rate (Default: 1e-3)
4. --wd          - Weight Decay (Default: 0)
5. --scheduler   - Needs two arguments; patience and eps
6. --epochs      - Number of training epochs (Default: 10)
7. --early       - Early stopping patience (Default: 5)
8. --model-name  - Name of the model (Supports 'resnet', 'vgg' & 'mobilenet')
9. --path        - Dataset Path (Default: None)
10. --augment    - Flag that controls train set augmentation (Default: None)
11. --reduce     - Flag that uses a reduced version of the dataset (Default: None)
12. --pretrained - Flag that controls whether to use a pretrained model (Default: None)
13. --test       - Flag that controls entry into test mode (Default: False)
14. --name       - Name of the image file to be tested (Default: Image_1.png)
</pre>

&nbsp;

---