import pandas as pd
import ast
import re
import torch.utils
import torch.utils.data
# import clip_helper
import cv2
import torch
import albumentations as A
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch.nn.functional as F
from tqdm import tqdm
# from viettext_processing import vietnamese_preprocessing
import itertools
import clip_helper
import viettext_processing
import numpy as np

# title - img
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, keywords, tokenizer, transforms):
        self.img_paths = img_paths
        self.keywords = list(keywords)
        print('CLIP DS flag')
        # print(keywords[0])
        # print(len(self.keywords))
        # print(type(self.keywords))
        
        self.encoded_keywords = tokenizer(
            self.keywords,  # List of texts to be tokenized
            padding=True,   # Pad the sequences to the maximum length
            truncation=True, # Truncate sequences to the maximum length
            max_length=clip_helper.CFG.max_length, # Maximum length for the sequences
            return_tensors='pt' # Return PyTorch tensors
        )# return_tensors='pt' # Return PyTorch tensors
        self.transforms = transforms
    
    def __getitem__(self, idx):
        item = {
            key: values[idx].clone().detach()
            for key, values in self.encoded_keywords.items()
        }
        # print('CLIP DS item flag')
        # print(item)
        image = cv2.imread(f"data/product_image/{self.img_paths[idx]}") # {clip_helper.CFG.image_path}/
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = item['image'] = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        item['caption'] = self.keywords[idx]
        return item
    
    def __len__(self):
        return len(self.keywords)
    

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(clip_helper.CFG.size, clip_helper.CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(clip_helper.CFG.size, clip_helper.CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=clip_helper.CFG.model_name, pretrained=clip_helper.CFG.pretrained, trainable=clip_helper.CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)



class TextEncoder(nn.Module):
    def __init__(self, model_name=clip_helper.CFG.text_encoder_model, pretrained=clip_helper.CFG.pretrained, trainable=clip_helper.CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=clip_helper.CFG.projection_dim,
        dropout=clip_helper.CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=clip_helper.CFG.temperature,
        image_embedding=clip_helper.CFG.image_embedding,
        text_embedding=clip_helper.CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def extract_values_from_string(string_data):
    try:
        data_list = re.sub(r'<.*?>', '', string_data)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid input string: {e}")
    return data_list

    
def prepare_data(data):
    cols = ['id', 'img_path', 'name']
    data = data[cols]
    img_paths = []
    keywords = []
    pids = []
    unique_entries = set()  # Set to track unique entries

    for pair in data.iterrows():
        pair = dict(pair[1])
        name = viettext_processing.vietnamese_preprocessing(pair.get('name'))
        img_paths.append(pair.get('img_path'))
        keywords.append(name)
        pids.append(pair.get('id'))
            
    return img_paths, keywords, pids

import regex as re

def prepare_data2(df):
    cols = ['id', 'img_path', 'keywords']
    df = df[cols]
    img_paths = []
    kws = []
    pids = []
    for _, row in df.iterrows():
        keywords = row.get('keywords')
        path = row.get('img_path')
        pid = row.get('id')
        keywords = keywords.split('-')
        keywords = [kw for kws in keywords for kw in kws.split('[SEP]') if not re.search(r'<[^>]+>', kw)]
        for kw in keywords:
            kws.append(kw)
            img_paths.append(path)
            pids.append(pid)
    return img_paths, kws, pids
# if not re.search(r'<[^>]+>', element)

def make_train_dfs():
    df = pd.read_csv('data/pdata/sample_imgs.csv')
    img_paths, keywords, pids = prepare_data2(df)
    dataset_df = pd.DataFrame({
        'pids':pids,
        'img_paths':img_paths,
        'keywords':keywords
    })
    dataset_df=dataset_df.drop_duplicates().reset_index().drop(columns=['index'])
    #print(dataset_df)

    # Split proportions
    train_frac = 0.9
    valid_frac = 0.1
    # test_frac = 0.05

    # Shuffle the dataset
    dataset_df = dataset_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate split sizes
    n = len(dataset_df)
    n_train = int(train_frac * n)
    n_valid = int(valid_frac * n)
    # n_test = n - n_train - n_valid

    # Split the dataset
    train_df = dataset_df.iloc[:n_train]
    valid_test_df = dataset_df.iloc[n_train:]

    # Shuffle valid_test_df and split into validation and test sets
    valid_test_df = valid_test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = valid_test_df.iloc[:n_valid]
    # test_df = valid_test_df.iloc[n_valid:]

    return dataset_df, train_df, valid_df

def build_loader(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    print('build loader df flag')
    # print(dataframe['keywords'].values)
    # print(len(dataframe['keywords'].values))
    
    dataset = CLIPDataset(
        img_paths=dataframe['img_paths'].values,
        keywords=dataframe['keywords'].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    print('build loader after CLIP DS build flag')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=clip_helper.CFG.batch_size,
        num_workers=clip_helper.CFG.num_workers,
        shuffle=True if mode =='Train' else False
    )

    print('build loader after torch dataloader flag')
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler,step):
    loss_meter=clip_helper.AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k:v for k,v in batch.items() if k!='keywords'}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if step == 'batch':
            lr_scheduler.step()
        count = batch['image'].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=clip_helper.get_lr(optimizer))
    return loss_meter

def valid_epoch(model, valid_loader):
    loss_meter = clip_helper.AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v for k, v in batch.items() if k != "keywords"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def testing_loaders(dataloader):
    print('build loader test_dataloader flag')
    print(dataloader)
    # print(help(dataloader))
    for batch_idx, (img_paths, keywords) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        for ip, kw in zip(img_paths, keywords):
            print(f"Keyword[0]: {kw[0]}")
            print(f"img path: {ip}")
            # Break after a few batches to avoid too much output
            break

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(clip_helper.CFG.text_tokenizer)
    valid_loader = build_loader(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(clip_helper.CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=clip_helper.CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    valid_text_embedidngs = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(clip_helper.CFG.device))
            text_features = model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]) # .to(clip_helper.clip_helper.CFG.device)
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)
            valid_text_embedidngs.append(text_embeddings)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings), torch.cat(valid_text_embedidngs)


def train(train_df, valid_df):
    print(train_df)

    tokenizer = DistilBertTokenizer.from_pretrained(clip_helper.CFG.text_tokenizer)
    print('===========')
    print('Train loader initializing...')
    print('===========')
    train_loader = build_loader(train_df, tokenizer, mode='train')
    print('===========')
    print('Valid loader initializing...')
    print('===========')
    valid_loader = build_loader(valid_df, tokenizer, mode='valid')

    # testing_loaders(train_loader)
    # testing_loaders(test_loader)
    # testing_loaders(valid_loader)
    print('\n===========')
    print('CLIP model initializing...')
    print('===========\n')
    model = CLIPModel().to(clip_helper.CFG.device)
    
    print('\n===========')
    print('CLIP params initializing...')
    print('===========\n')
    params = [
        {"params": model.image_encoder.parameters(), "lr":clip_helper.CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr":clip_helper.CFG.text_encoder_lr},
        {"params":itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr":clip_helper.CFG.head_lr, "weight_decay":clip_helper.CFG.weight_decay}
    ]
    print('\n===========')
    print('CLIP optimizer initializing...')
    print('===========\n')
    optimizer = torch.optim.AdamW(params=params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", patience=clip_helper.CFG.patience,
        factor=clip_helper.CFG.factor
    )
    step = 'epoch'

    print('\n===========')
    print('CLIP training process started...')
    print('===========\n')
    best_loss = float('inf')
    for epoch in range(clip_helper.CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "model/best_og.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

def test(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_og.pt')
    print(img_embeddings.shape)
    print(text_embeddings.shape)

    df_img = pd.DataFrame({
        'id': ds_df['pids'],
        'name':ds_df['keywords'],
        'img_path':ds_df['img_paths'],
        'img_embeddings':list(img_embeddings),
        'text_embeddings':list(text_embeddings)
    })
    df_img.to_csv('embeddings/clip_img_embedding.csv', encoding='utf-8', index=False)


def main():
    ds_df, train_df, valid_df = make_train_dfs()
    print(ds_df)
    #train(train_df, valid_df)
    test(ds_df)
    

# Run the main function
if __name__ == "__main__":
    main()