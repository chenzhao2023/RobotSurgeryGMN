**Abstract:-**\
We propose a novel graph-based framework for automatic surgical report generation that integrates graph similarity and contextual embedding techniques. Our model leverages SimGNN, a graph matching algorithm, to compute similarity scores between an input surgical scene graph and a set of template graphs annotated with captions. The input graph is concurrently processed by a Graph Attention Network (GAT), composed of two attention layers followed by a fully connected network (FCN). The FCN aggregates the GAT-derived embeddings and the caption embeddings from the top-k most similar template graphs (as determined by SimGNN), aligning them to the input dimensions of a pre-trained BERT model. These fused embeddings are then used to generate semantically rich captions through beam search decoding. Our architecture effectively combines structural similarity and learned semantic representations, enabling accurate and contextually appropriate report generation for robotic surgery scenarios.

**Data Files:**

Download the required json files, instruments_captions and annotation files from the link :-
<a>https://kennesawedu-my.sharepoint.com/personal/czhao4_kennesaw_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fczhao4%5Fkennesaw%5Fedu%2FDocuments%2FResearch%2FAkshay%2FRobotSurgeryGMN&e=5%3Acd7ff596cdf04d7fbfbde02967a0c850&sharingv2=true&fromShare=true&at=9&CID=f68b631d%2Db403%2D4dc2%2D8338%2D1d7a35a9c2a3&FolderCTID=0x012000351DFEEC48B9C248BBA0D2BE4531A027&view=0</a>




**Model training:-**

**Command for model training:**
python GMN.py \
  --template_file path to template json file \
  --train_file path to captions_train json file \
  --val_file file path to captions_val json file \
  --save_path model save path \
  --batch_size 4 \
  --epochs 100 \
  --learning_rate 1e-3 \
  --caption_learning_rate 1e-5 \
  --use_geometric \
  --beam_width 3 \
  --max_length 50 \
  --primary_metric ROUGE_L \
  --weight_decay 1e-5 \
  --max_nodes 6 \
  --base_path base path to instruments18_caption directory

**Command for model evaluation:**

  python GMN.py \
  --template_file path to template json file \
  --train_file path to captions_train json file \
  --val_file file path to captions_val json file \
  --save_path model save path \
  --batch_size 4 \
  --beam_width 3 \
  --max_length 50 \
  --eval_only \
  --use_geometric \
  --max_nodes 6 \
  --base_path base path to instruments18_caption directory

**Requirements:**
**Environment Requirements**
  
  Python >= 3.8 (Python 3.9+ recommended for latest PyTorch)\
  CUDA compatible GPU (recommended for training)
  
**Main Dependencies**
  torch>=1.9.0\
  transformers>=4.12.0\
  torch-geometric>=2.3.0\
  numpy>=1.19.0\
  matplotlib>=3.3.0\
  tqdm>=4.50.0\
  nltk>=3.5\
**PyTorch Geometric Dependencies**
  PyTorch Geometric (PyG) requires additional dependencies for full functionality:\
  pyg-lib\
  torch-scatter\
  torch-sparse\
