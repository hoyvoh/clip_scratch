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

class CLIP_RDF_Model(CLIPModel):
    def __init__(
            self,
            temperature = clip_helper.CFG.temperature,
            image_embedding=clip_helper.CFG.image_embedding,
            text_embedding=clip_helper.CFG.text_embedding,
            margin=0.2,
            beta=0.5
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.margin = margin
        self.beta = beta

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
        logits = cosine_similarity(text_embeddings, image_embeddings)/self.temperature - self.margin
        text_similarity = cosine_similarity(text_embeddings, text_embeddings)
        image_similarity = cosine_similarity(image_embeddings, image_embeddings)
        targets = (image_similarity + text_similarity)/2*self.temperature
        texts_loss = cross_entropy(logits, targets, reduction='none') * self.beta
        images_loss = cross_entropy(logits.T, targets.T, reduction='none') * self.beta
        loss = (images_loss + texts_loss)/2
        return loss.mean()


def train_rdf(train_df, valid_df):
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
    model = CLIP_RDF_Model().to(clip_helper.CFG.device)
    
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
            torch.save(model.state_dict(), "model/best_rfd.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

def test_rfd(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_rfd.pt')
    print(img_embeddings.shape)
    print(text_embeddings.shape)

    df_img = pd.DataFrame({
        'id': ds_df['pids'],
        'name':ds_df['keywords'],
        'img_path':ds_df['img_paths'],
        'img_embeddings':list(img_embeddings),
        'text_embeddings':list(text_embeddings)
    })
    df_img.to_csv('embeddings/clip_img_embedding_rfd.csv', encoding='utf-8', index=False)

def main_rdf():
    df, train_df, valid_df = make_train_dfs()
    print(df)
    train_rdf(train_df, valid_df)
    test_rfd(df)

    

if __name__ == '__main__':
    main_rdf()