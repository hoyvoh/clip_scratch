from clip_base import *
import random

'''
RDML:
- Init M as identity matrix
- for each step, take in xi, xj and yij is always  1
- for img xi, take n-3 xk for k not j  from the list of captions then update yij in this case is always -1
- for each pair of xi, xj, calculate loss and yield return loss for update
- New M updated using the formula based on xi, xj, yij
- Lambda: trade off rate
- setup the dataset so that 1 img has 3 to 5 keywords and no duplicate,
then, for each keyword, pair with img not its class with a sign -1


'''
torch.autograd.set_detect_anomaly(True)


def prepare_rdml_data(df):
    cols = ['id', 'img_path', 'keywords']
    df = df[cols]
    img_paths = []
    kws = []
    pids = []
    signals = []

    all_keywords = []
    for keywords in df['keywords']:
        keywords = keywords.split('-')
        keywords = [kw for kws in keywords for kw in kws.split('[SEP]') if not re.search(r'<[^>]+>', kw)]
        all_keywords.extend(keywords)

    for _, row in df.iterrows():
        keywords = row.get('keywords')
        path = row.get('img_path')
        pid = row.get('id')
        keywords = keywords.split('-')
        keywords = [kw for kws in keywords for kw in kws.split('[SEP]') if not re.search(r'<[^>]+>', kw)]

        # Positive samples
        for kw in keywords:
            kws.append(kw)
            img_paths.append(path)
            pids.append(pid)
            signals.append(1.0)  # Positive signal

        # Negative samples
        for kw in random.sample(all_keywords, len(keywords)):
            if kw not in keywords:
                kws.append(kw)
                img_paths.append(path)
                pids.append(pid)
                signals.append(-1.0)  # Negative signal

    return img_paths, kws, pids, signals
    

def make_rdml_train():
    df = pd.read_csv('data/pdata/sample_imgs.csv')
    img_paths, keywords, pids, signals = prepare_rdml_data(df)
    dataset_df = pd.DataFrame({
        'pids':pids,
        'img_paths':img_paths,
        'keywords':keywords,
        'signals':signals
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


class CLIP_RDML_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, keywords, signals, tokenizer, transforms):
        self.img_paths = img_paths
        self.keywords = list(keywords)
        self.signals = list(signals)
        self.encoded_keywords = tokenizer(
            self.keywords, 
            padding=True,
            truncation=True,
            max_length=clip_helper.CFG.max_length,
            return_tensors='pt'
        )
        self.transforms=transforms
    
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
        item['signal'] = self.signals[idx]
        return item
    
    def __len__(self):
        return len(self.keywords)
        

def build_RDML_loader(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    print('build loader df flag')
    # print(dataframe['keywords'].values)
    # print(len(dataframe['keywords'].values))
    
    dataset = CLIP_RDML_Dataset(
        img_paths=dataframe['img_paths'].values,
        keywords=dataframe['keywords'].values,
        signals=dataframe['signals'].values,
        tokenizer=tokenizer,
        transforms=transforms
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


import torch.nn.functional as F
import torch.nn as nn

def cosine_similarity(x, y):
    x1 = x/x.norm(dim=-1, keepdim=True)
    y1 = y/y.norm(dim=-1, keepdim=True)
    similarity = x1 @ y1.T
    return similarity

class CLIP_RDML_Model(CLIPModel):
    def __init__(
            self,
            temperature = clip_helper.CFG.temperature,
            image_embedding=clip_helper.CFG.image_embedding,
            text_embedding=clip_helper.CFG.text_embedding,
            
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.model_lambda = clip_helper.CFG.model_lambda
        self.mahalanobis_matrix = torch.eye(n=clip_helper.CFG.projection_dim)

    def mahalanobis_similarity(self, x1, x2):
        diff = x1 - x2
        distance = torch.sqrt(torch.matmul(torch.matmul(diff, self.mahalanobis_matrix), diff.T))
        similarity = 1 / (1 + distance)
        return similarity

    def update_matrix(self, x1, x2, y):
        diff = x1 - x2
        #y_expanded = y.unsqueeze(1).expand_as(diff)
        weighted_loss_matrix = torch.matmul(y.to(torch.float),torch.matmul(diff, diff.T))
        weighted_loss = self.model_lambda * torch.sum(weighted_loss_matrix)
        self.mahalanobis_matrix = self.mahalanobis_matrix - weighted_loss
        self.mahalanobis_matrix = self.project_to_psd(self.mahalanobis_matrix)

    def project_to_psd(self, matrix):
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=0)
        return eigvecs @ torch.diag(eigvals) @ eigvecs.T

    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'], attention_mask=batch["attention_mask"]
        )
        signals = batch['signal']

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculate similarities
        img_text_similarity = self.mahalanobis_similarity(image_embeddings, text_embeddings)
        text_text_similarity = self.mahalanobis_similarity(text_embeddings, text_embeddings)
        img_img_similarity = self.mahalanobis_similarity(image_embeddings, image_embeddings)

        # Targets
        targets = (img_img_similarity + text_text_similarity) / 2 * self.temperature

        # Calculate logits
        logits = img_text_similarity / self.temperature

        # Loss calculation
        texts_loss = F.cross_entropy(logits, targets, reduction='none') * signals
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none') * signals

        loss = (images_loss.mean() + texts_loss.mean()) / 2

        # Update Mahalanobis matrix
        self.update_matrix(image_embeddings, text_embeddings, signals)
        print(loss)

        return loss


def train_rdml(train_df, valid_df):
    tokenizer = DistilBertTokenizer.from_pretrained(clip_helper.CFG.text_tokenizer)
    print('===========')
    print('Train loader initializing...')
    print('===========')
    train_loader = build_RDML_loader(train_df, tokenizer, mode='train')
    print('===========')
    print('Valid loader initializing...')
    print('===========')
    valid_loader = build_RDML_loader(valid_df, tokenizer, mode='valid')

    # testing_loaders(train_loader)
    # testing_loaders(test_loader)
    # testing_loaders(valid_loader)
    print('\n===========')
    print('CLIP model initializing...')
    print('===========\n')
    model = CLIP_RDML_Model().to(clip_helper.CFG.device)
    
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
            torch.save(model.state_dict(), "model/best_rdml.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

def test_rdml(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_rdml.pt')
    print(img_embeddings.shape)
    print(text_embeddings.shape)

    df_img = pd.DataFrame({
        'id': ds_df['pids'],
        'name':ds_df['keywords'],
        'img_path':ds_df['img_paths'],
        'img_embeddings':list(img_embeddings),
        'text_embeddings':list(text_embeddings)
    })
    df_img.to_csv('embeddings/clip_img_embedding_rdml.csv', encoding='utf-8', index=False)

def main_rdml():
    df, train_df, valid_df = make_rdml_train()
    print(df)
    train_rdml(train_df, valid_df)
    test_rdml(df)

def function_tester():
    print(make_rdml_train())
    

if __name__ == '__main__':
    main_rdml()