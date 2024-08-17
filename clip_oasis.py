from clip_base import *


'''
CLIPoasisModel(CLIPModel):


'''

import torch
import torch.nn as nn

class BilinearSimilarity(nn.Module):
    def __init__(self, embedding_dim):
        super(BilinearSimilarity, self).__init__()
        # Initialize the learnable matrix M (identity matrix as the starting point)
        self.M = nn.Parameter(torch.eye(embedding_dim))
    
    def forward(self, x1, x2):
        # Compute bilinear similarity KM(x1, x2) = x1^T * M * x2
        return torch.matmul(torch.matmul(x1, self.M), x2.T)


class CLIPoasisModel(CLIPModel):
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
        self.bilinear_similarity = BilinearSimilarity(clip_helper.CFG.projection_dim)

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
        logits = self.bilinear_similarity(image_embeddings, text_embeddings)/self.temperature
        


def train_oasis():
    pass

def main():
    df, train_df, valid_df = make_train_dfs()
    
    

if __name__ == '__main__':
    main()