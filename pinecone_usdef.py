from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
import clip_helper
import regex as re
import pandas as pd
import torch
from transformers import DistilBertTokenizer
    
example_df = pd.read_csv('data/pdata/sample_imgs.csv') 

class PineconeClientAccess():
    def __init__(self, name='groceries-tiki'):
        self.index_name = name
        self.api_key = "47e79840-2174-4296-82fc-45212d1f190f"
        self.pineconce = Pinecone(api_key=self.api_key)
        
        if self.index_name not in self.pineconce.list_indexes().names():
            self.pineconce.create_index(
                name=self.index_name,
                dimension=clip_helper.CFG.projection_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'
                ) 
            )
        self.index = self.pineconce.Index(self.index_name)
    
    def upsert_namespace(
            self,
            text_embeddings,
            image_embeddings,
            pids,
            metadata,
            namespace
    ):
        vectors = []
        ids = [i for i in range(len(pids))]
        
        metadata_list = []
        for meta, pid in zip(metadata, pids):
            meta_dict = {}
            meta_el = meta.split('-')
            meta_el = [kw for kws in meta_el for kw in kws.split('[SEP]') if not re.search(r'<[^>]+>', kw)]
            for pair in meta_el:
                key_value = pair.split(':', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    if key in meta_dict:
                        if isinstance(meta_dict[key], list):
                            meta_dict[key].append(value)
                        else:
                            meta_dict[key] = [meta_dict[key], value]
                    else:
                        meta_dict[key] = value
                else: 
                    meta_dict['sản phẩm'] = key_value
            meta_dict['mã sản phẩm'] = pid
            metadata_list.append(meta_dict)

        for id, text_embedding, image_embedding, meta in zip(ids, text_embeddings, image_embeddings, metadata_list):
            meta_text = meta
            meta_img = meta
            meta_img['type'] = 'image'
            meta_text['type'] = 'text'

            vector_text = {
                "id":str(id)+'_t', 
                "values":text_embedding,
                "metadata":meta_text
            }

            vector_img = {
                "id": str(id)+'_i',
                "values":image_embedding,
                "metadata":meta_img
            }

            vectors.append(vector_text)
            vectors.append(vector_img)

        self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )

    def namespace_details(self, namespace):
        print(self.index.describe_index_stats()["namespaces"][namespace])

    def query_namespace(self, namespace, query):
        encoded_query = self.encode_query(query)
        response = self.index.query(
            namespace=namespace,
            vector=encoded_query.flatten().tolist(),
            top_k=9,
            include_metadata=True,
            include_values=True
        )
        return response

    def encode_query(query_text, model_path, CLIPModelObject):
        tokenizer = DistilBertTokenizer.from_pretrained(clip_helper.CFG.text_tokenizer)
        model = CLIPModelObject().to(clip_helper.CFG.device)
        model.load_state_dict(torch.load(model_path, map_location=clip_helper.CFG.device))
        model.eval()
        encoded_input = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)

        # Step 3: Generate query embedding
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=encoded_input["input_ids"].to(clip_helper.CFG.device),
                attention_mask=encoded_input["attention_mask"].to(clip_helper.CFG.device)
            )
            query_embedding = model.text_projection(text_features)

        return query_embedding.detach().cpu().numpy()        
    
    

         
