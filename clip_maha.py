from clip_base import *
import torch
import torch.nn as nn
from mahalanobis_package import MahalanobisScore, MarginalMahalanobisScore  

'''
CLIPMahasModel(CLIPModel):


'''
torch.autograd.set_detect_anomaly(True)

class MahalanobisDistance:
    def __init__(self, dim, regularization=1e-6, ema_alpha=0.01):
        self.dim = dim
        self.regularization = regularization
        self.ema_alpha = ema_alpha
        self.covariance_matrix = torch.eye(dim, dtype=torch.float)
        self.mean_vector = torch.zeros(dim, dtype=torch.float)
        self.total_samples = 0

    def update(self, embeddings):
        batch_size = embeddings.size(0)
        batch_mean = embeddings.mean(dim=0)
        centered_embeddings = embeddings - batch_mean
        batch_covariance = centered_embeddings.T @ centered_embeddings/batch_size
        self.covariance_matrix = (1-self.ema_alpha)*self.covariance_matrix + self.ema_alpha * batch_covariance
        self.mean_vector = (self.total_samples * self.mean_vector + batch_size * batch_mean) / (self.total_samples + batch_size)
        self.total_samples += batch_size

    def finalize_covariance(self):
        self.covariance_matrix += self.regularization * torch.eye(self.dim)
        self.covariance_matrix = self.covariance_matrix.inverse()
    
    def compute_distance(self, embeddings):
        centered_embeddings = embeddings - self.mean_vector
        mahalanobis_distance = torch.diag(centered_embeddings @ self.covariance_matrix @ centered_embeddings.T)
        print('maha distance shape: ', mahalanobis_distance.shape)
        return mahalanobis_distance

    def compute_targets(self, text, img):
        trace = torch.trace(self.covariance_matrix)
        normalized_matrix = self.covariance_matrix / trace

        text_sim = text @ normalized_matrix @ text.T 
        img_sim = img @ normalized_matrix @ img.T
        targets = F.softmax(
            (text_sim+img_sim)/(2), dim=-1
        )
        return targets
    
    def distance_to_similarity(self, distance):
        return torch.exp(-distance)


class CLIP_Maha_Model(CLIPModel):
    def __init__(
            self,
            temperature = clip_helper.CFG.temperature,
            image_embedding=clip_helper.CFG.image_embedding,
            text_embedding=clip_helper.CFG.text_embedding
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.mahalanobis_distance = MahalanobisDistance(dim=clip_helper.CFG.projection_dim)
        

    def forward(self, batch):
        image_features = self.image_encoder(batch['image'])
        text_features = self.text_encoder(
            input_ids=batch['input_ids'], attention_mask=batch["attention_mask"]
        )
        print('ITERATION REPORT')
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Update mahalanobis covar matrix
        sub_embedding = image_embeddings - text_embeddings
        self.mahalanobis_distance.update(sub_embedding)
        print('===================')
        print('sub embedding:', sub_embedding)
        print('covar matrix:', self.mahalanobis_distance.covariance_matrix)
        print('covar matrix shape:', self.mahalanobis_distance.covariance_matrix.shape)
        print('===================')

        img_distance = self.mahalanobis_distance.compute_distance(image_embeddings)
        text_distance = self.mahalanobis_distance.compute_distance(text_embeddings)
        text_img_distances = self.mahalanobis_distance.compute_distance(sub_embedding)
        print('===================')
        print('image distance: ', img_distance)
        print('text distance: ', text_distance)
        print('image text distance: ', text_img_distances)
        print('===================')

        # img_similarity = self.mahalanobis_distance.distance_to_similarity(img_distance)
        # text_similarity = self.mahalanobis_distance.distance_to_similarity(text_distance)
        # text_img_similarity = self.mahalanobis_distance.distance_to_similarity(text_img_distances)
        # print('===================')
        # print('img similarity: ', img_similarity)
        # print('text similarity: ', text_similarity)
        # print('img text similarity: ', text_img_similarity)
        # print('===================')

        logits = F.softmax((img_distance+text_distance+text_img_distances)/(3*self.temperature), dim=-1) # [32, ]
        targets = torch.zeros(size=(clip_helper.CFG.batch_size,))

        print('===================')
        print('logits: ', logits)
        print('logits: ', logits.shape)
        print('targets: ', targets)
        print('targets: ', targets.shape)
        print('===================')

        text_loss = F.mse_loss(logits, targets, reduction='none')
        image_loss = F.mse_loss(logits, targets, reduction='none')
        loss = (image_loss+text_loss)/2.0
        
        print('===================')
        print('text loss: ', text_loss)
        print('image loss', image_loss)
        print('Train loss: ', loss)
        print('===================')

        return loss.mean()

def train_maha(train_df, valid_df):
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
    model = CLIP_Maha_Model().to(clip_helper.CFG.device)
    
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
            torch.save(model.state_dict(), "model/best_maha.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

def test_maha(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_maha.pt')
    print(img_embeddings.shape)
    print(text_embeddings.shape)

    df_img = pd.DataFrame({
        'id': ds_df['pids'],
        'name':ds_df['names'],
        'img_path':ds_df['img_paths'],
        'img_embeddings':list(img_embeddings),
        'text_embeddings':list(text_embeddings)
    })

    pc_client.upsert_namespace(
        text_embeddings=list(text_embeddings), 
        image_embeddings=list(img_embeddings),
        pids=ds_df['pids'].tolist(),
        metadata=ds_df['keywords'].tolist(),
        namespace='mahaCLIP'
    )
    pc_client.namespace_details(namespace='mahaCLIP')

    df_img.to_csv('embeddings/clip_img_embedding_maha.csv', encoding='utf-8', index=False)

def main_maha():
    df, train_df, valid_df = make_train_dfs()
    print(df)
    train_maha(train_df, valid_df)
    test_maha(df)
    

if __name__ == '__main__':
    main_maha()