import pandas as pd
import torch
import faiss
import time
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

def prepsearch(norm=True, df = None, target = 'content', importing = False):
    ins = SearchEngine(target=target)
    if df is None:
        ins.importcsv("articles.csv")
    else:
        ins.importdf(df)
    if importing:
        ins.importencoded()
    ins.buildindex()
    if norm:
        ins.em = np.array(ins.em).astype("float32")
        ins.normalizeencoded()
    return ins

def summary_prepsearch():
    ins = SearchEngine(target="text")
    ins.importcsv("articles_with_summary.csv")
    ins.importencoded(path="embs_with_summary.npy")
    ins.buildindex()
    return ins

class SearchEngine():
    def __init__(self,  
            pretrained = "distilbert-base-nli-stsb-mean-tokens",
            target = "content"):
        self.encoder = SentenceTransformer(pretrained)
        self.target = target
        self.em = None
        if torch.cuda.is_available():
            self.encoder = self.encoder.to(torch.device("cuda"))
    
    def importcsv(self, path):
        self.df = pd.read_csv(path)

    def importdf(self, df):
        self.df = df

    def normalizeencoded(self):
        self.em = normalize(self.em)

    def importencoded(self, path="embeddings_DistilBert.npy"):
        try:
            self.em = np.load(path)
            self.vecdim = self.em.shape[1]
            print("Encoded text database imported successfully")
        except:
            print("ERROR: CANNOT import encoded text database")

    def buildindex(self ):
        try:
            if self.em is None:
                print("No imported encoded text database.")
                dec = input("Would you like to encode? (it may take ~ 1 hour)\n (y/n): ")
                if dec.lower()[0] == 'y':
                    self.em = self.encoder.encode(self.df[self.target].to_list(),show_progress_bar=True)
                    self.em = np.array([emi for emi in self.em]).astype("float32")
                    self.vecdim = self.em.shape[1]
                else:
                    path = input("Enter the path to encoded text base: ")
                    self.importencoded(path)
            #self.index = faiss.IndexFlatL2(self.vecdim)
            self.index = faiss.IndexFlatIP(self.vecdim)
            self.index = faiss.IndexIDMap(self.index)
            self.normalizeencoded()
            self.index.add_with_ids(self.em, self.df.id.values)
            print("FAISS index was built successfully")
            print("Number of articles:", self.index.ntotal)
        except:
            print("ERROR: CANNOT build index")
    
    def searchquery(self, text_query, k=5, to_display = ["title"], export_txt=False, display_content=False):
        tic = time.time()
        vector_query = self.encoder.encode(list([text_query]))
        vector_query = np.array(vector_query).astype("float32")
        vector_query = normalize(vector_query)
        Dists, Ids = self.index.search(vector_query, k = k)
        reslist = [self.df[self.df.id == idx][to_display].values for idx in Ids[0]]
        print("Time taken for search is {}".format(time.time() - tic))
        for k,i in enumerate(reslist):
            print("rank\t: ",k+1)
            print("metric\t: ", Dists[0][k])
            for j,target in enumerate(to_display):
                print(target,"\t: ",i[0][j])
            print()
        if export_txt:
            with open(f"{text_query}.txt", "w") as f:
                f.write("Time taken for search is {}".format(time.time() - tic))
                for k,i in enumerate(reslist):
                    f.write("rank\t: ",k+1)
                    f.write("metric\t: ", Dists[0][k])
                    if display_content:
                        to_display = ["title", "content"]
                    for j,target in enumerate(to_display):
                        f.write(target,"\t: ",i[0][j])
                    f.write()
            f.close()


    def saveencoded(self, path = "embs"):
        np.save(path, self.em)
