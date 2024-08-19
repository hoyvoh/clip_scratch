from clip_base import *
import random

'''
RDML:
Data: sample data in 2 ways: 
> Positive: img - title - signal = 1
> Negarive: img - title - signal = -1 x 5

- M is initialized as identity
- learning rate = 1e-4
- projector to PSD only for M


'''


def prepare_rdml_data(df):
    cols = ['id', 'img_path', 'name','keywords']
    df = df[cols]
    img_paths = []
    keywords = []
    names = []
    pids = []
    signals = []
    negative_labels = df['name'].tolist()
    
    for _, row in df.iterrows():
        name = viettext_processing.vietnamese_preprocessing(row.get('name'))
        
        # positive line
        img_paths.append(row.get('img_path'))
        names.append(name)
        keywords.append(row.get('keywords'))
        pids.append(row.get('id'))
        signals.append(1.0)

        # negative lines
        dup_mems = set(name.split(' '))
        available_subset = [
            label for label in negative_labels
            if label != name and not dup_mems.intersection(label.lower().split())
        ]
        if len(available_subset) >3:
            sampled_products = random.sample(available_subset, 3)
        else:
            sampled_products = available_subset
        for neg in sampled_products:
            img_paths.append(row.get('img_path'))
            names.append(neg)
            keywords.append(row.get('keywords'))
            pids.append(row.get('id'))
            signals.append(-1.0)
    return img_paths, names, pids, keywords, signals


def make_rdml_train():
    df = pd.read_csv('data/pdata/sample_imgs.csv')
    img_paths, names, pids, keywords, signals = prepare_rdml_data(df)
    dataset_df = pd.DataFrame({
        'pids':pids,
        'img_paths':img_paths,
        'names': names,
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
    def __init__(self, img_paths, names, signals, tokenizer, transforms):
        self.img_paths = img_paths
        self.names = list(names)
        self.signals = list(signals)
        self.encoded_keywords = tokenizer(
            self.names, 
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
        item['caption'] = self.names[idx]
        item['signal'] = self.signals[idx]
        return item
    
    def __len__(self):
        return len(self.names)
        

def build_RDML_loader(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    print('build loader df flag')
    # print(dataframe['keywords'].values)
    # print(len(dataframe['keywords'].values))
    
    dataset = CLIP_RDML_Dataset(
        img_paths=dataframe['img_paths'].values,
        names=dataframe['names'].values,
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

torch.autograd.set_detect_anomaly(True)

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
        distance = torch.sqrt(torch.clamp(torch.matmul(torch.matmul(diff, self.mahalanobis_matrix), diff.T), min=1e-6, max=1e3))
        similarity = 1 / (1 + distance)
        return similarity

    def update_matrix(self, x1, x2, y):
        diff = x1 - x2
        y = y.to(torch.float)
        weighted_loss_matrix = torch.clamp(torch.matmul(y, torch.matmul(diff, diff.T)), min=1e-6, max=1e2)
        new_matrix = self.mahalanobis_matrix - self.model_lambda * weighted_loss_matrix.mean(dim=0)
        # regularization
        epsilon = 1e-6
        new_matrix += epsilon * torch.eye(new_matrix.size(0), device=new_matrix.device)

        self.mahalanobis_matrix = self.project_to_psd(new_matrix)

    def project_to_psd(self, matrix):
        print(matrix)
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        eigvals = torch.clamp(eigvals, min=1e-6)
        psd_matrix = torch.clamp(eigvecs @ torch.diag(eigvals) @ eigvecs.T, min=1e-6, max=1e2)
        print(psd_matrix)
        return psd_matrix

    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'], attention_mask=batch["attention_mask"]
        )
        signals = batch['signal']
        print('signal:', signals.shape)

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        print('----------------')
        print('img:', image_embeddings.shape)
        print('text:', text_embeddings.shape)        
        print('----------------')

        # Calculate similarities
        img_text_similarity = self.mahalanobis_similarity(image_embeddings, text_embeddings)
        text_text_similarity = self.mahalanobis_similarity(text_embeddings, text_embeddings)
        img_img_similarity = self.mahalanobis_similarity(image_embeddings, image_embeddings)
        print('----------------')
        print('img-text:', img_text_similarity.shape)
        print('img-img:', img_img_similarity.shape)
        print('text-text:', text_text_similarity.shape)
        print('----------------')
        # Targets
        targets = torch.clamp((img_img_similarity + text_text_similarity) / 2 * self.temperature, min=1e-6, max=1e3)
        print('----------------')
        print('targets', targets)
        print('----------------')
        # Calculate logits
        logits = torch.clamp(img_text_similarity / self.temperature, min=1e-6, max=1e3)
        print('----------------')
        print('logits', logits)
        print('----------------')

        self.update_matrix(image_embeddings, text_embeddings, signals)
        signals = torch.tensor(signals)
        signals[signals == -1] = 0
        
        # Loss calculation
        texts_loss = F.cross_entropy(logits, targets, reduction='none') * signals
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none') * signals
        loss = (images_loss.mean() + texts_loss.mean()) / 2

        print('----------------')
        print('text loss', texts_loss)
        print('img loss', images_loss)
        print('loss', loss)
        print('----------------')

        # Update Mahalanobis matrix
        
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

    pc_client.upsert_namespace(
        text_embeddings=list(text_embeddings), 
        image_embeddings=list(img_embeddings),
        pids=ds_df['pids'].tolist(),
        metadata=ds_df['keywords'].tolist(),
        namespace='rdmlCLIP'
    )
    pc_client.namespace_details(namespace='rdmlCLIP')

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