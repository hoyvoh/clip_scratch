from clip_base import *


'''
CLIPoasisModel(CLIPModel):


'''

import torch.nn.functional as F
import torch.nn as nn

def cosine_similarity(x, y):
    x1 = x/x.norm(dim=-1, keepdim=True)
    y1 = y/y.norm(dim=-1, keepdim=True)
    similarity = torch.abs(x1 @ y1.T)
    return similarity

def euclid_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))

# # no
# def minkowski_distance(x, y):
#     if x.shape != y.shape:
#         raise ValueError("Vectors must have the same length")
#     return torch.sum(x != y).item()


def minkowski_distance(x, y, p=1):
    if x.shape != y.shape:
        raise ValueError("Vectors must have the same length")
    if p <= 0 or p > 2:
        p = 1
    distance = torch.sum(torch.abs(x - y) ** p, dim=1, keepdim=True).pow(1/p)
    distance_matrix = torch.diag(distance)
    return distance_matrix


class CLIP_minkowski_Model(CLIPModel):
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
        

    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'], attention_mask=batch["attention_mask"]
        )

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        '''
        [Optimal Affine Subspace for Image Similarity]
        Calculate the loss:
        - 
        
        '''
        logits = F.softmax(minkowski_distance(text_embeddings, image_embeddings)/self.temperature, dim=-1)
        text_distance = minkowski_distance(text_embeddings, text_embeddings)
        image_distance = minkowski_distance(image_embeddings, image_embeddings)
        targets = F.softmax((image_distance + text_distance)/2*self.temperature, dim=-1)
        texts_loss = F.mse_loss(logits, targets, reduction='none')
        images_loss = F.mse_loss(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss)/2
        return loss.mean()


def train_minkowski(train_df, valid_df):
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
    model = CLIP_minkowski_Model().to(clip_helper.CFG.device)
    
    print('\n===========')
    print('CLIP params initializing...')
    print('===========\n')
    params = [
        {"params": model.image_encoder.parameters(), "lr":clip_helper.CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr":clip_helper.CFG.image_encoder_lr},
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
            torch.save(model.state_dict(), "model/best_minkowski.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)


def test_minkowski(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_minkowski.pt')
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
        namespace='minkowskiCLIP'
    )
    pc_client.namespace_details(namespace='minkowskiCLIP')

    df_img.to_csv('embeddings/clip_img_embedding_minkowski.csv', encoding='utf-8', index=False)

def main_rdf():
    df, train_df, valid_df = make_train_dfs()
    print(df)
    train_minkowski(train_df, valid_df)
    test_minkowski(df)

    

if __name__ == '__main__':
    main_rdf()