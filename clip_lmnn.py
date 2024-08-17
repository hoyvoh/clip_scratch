from clip_base import *


'''
CLIPoasisModel(CLIPModel):


'''

import torch.nn.functional as F
import torch.nn as nn

def cosine_similarity(x, y):
    x1 = x/x.norm(dim=-1, keepdim=True)
    y1 = y/y.norm(dim=-1, keepdim=True)
    similarity = x1 @ y1.T
    return similarity

class CLIP_LMNN_Model(CLIPModel):
    def __init__(
        self,
        temperature=clip_helper.CFG.temperature,
        image_embedding=clip_helper.CFG.image_embedding,
        text_embedding=clip_helper.CFG.text_embedding,
        lmnn_margin=1.0,
        lmnn_alpha=0.5,
        embedding_dim=512,  # Example dimension
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
        self.lmnn_margin = lmnn_margin
        self.lmnn_alpha = lmnn_alpha

        # Learnable Mahalanobis matrix
        self.M = nn.Parameter(torch.eye(embedding_dim))  # Initialize as identity matrix

    def compute_mahalanobis_distance(self, x, y):
        """ Compute the Mahalanobis distance between x and y using the matrix M. """
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        distance = torch.sum(diff @ self.M @ diff.transpose(1, 2), dim=-1)
        return distance

    def lmnn_loss(self, embeddings, labels):
        """ Compute the LMNN loss. """
        batch_size = embeddings.size(0)
        distance_matrix = self.compute_mahalanobis_distance(embeddings, embeddings)
        labels = labels.unsqueeze(1)  # Make labels column vector

        # Pulling loss
        same_class_mask = labels == labels.T
        pulling_loss = (distance_matrix * same_class_mask).sum() / same_class_mask.sum()

        # Generate pairs for pushing loss
        push_constraints = (labels != labels.T).nonzero(as_tuple=True)
        push_mask = (labels != labels.T).float()
        pushing_loss = F.relu(self.lmnn_margin - distance_matrix[push_constraints]).sum() / push_mask.sum()

        # Combined loss
        loss = self.lmnn_alpha * pulling_loss + (1 - self.lmnn_alpha) * pushing_loss
        return loss

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Combine embeddings for loss calculation
        embeddings = torch.cat([image_embeddings, text_embeddings], dim=0)
        labels = torch.cat([batch["image_labels"], batch["text_labels"]], dim=0)  # Assuming you have labels for both

        # Calculating the LMNN Loss
        lmnn_loss_value = self.lmnn_loss(embeddings, labels)

        # Calculate the Contrastive Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = F.cross_entropy(logits, targets, reduction='none')
        images_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        contrastive_loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)

        # Combined loss
        total_loss = contrastive_loss.mean() + lmnn_loss_value

        return total_loss


def train_lmnn(train_df, valid_df):
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
    model = CLIP_LMNN_Model().to(clip_helper.CFG.device)
    
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
            torch.save(model.state_dict(), "model/best_lmnn.pt")
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

def test_lmnn(ds_df):
    _, img_embeddings, text_embeddings = get_image_embeddings(ds_df, 'model/best_lmnn.pt')
    print(img_embeddings.shape)
    print(text_embeddings.shape)

    df_img = pd.DataFrame({
        'id': ds_df['pids'],
        'name':ds_df['keywords'],
        'img_path':ds_df['img_paths'],
        'img_embeddings':list(img_embeddings),
        'text_embeddings':list(text_embeddings)
    })
    df_img.to_csv('embeddings/clip_img_embedding_lmnn.csv', encoding='utf-8', index=False)

def main_rdf():
    df, train_df, valid_df = make_train_dfs()
    print(df)
    train_lmnn(train_df, valid_df)
    test_lmnn(df)

    

if __name__ == '__main__':
    main_rdf()