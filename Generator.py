from keras.utils import Sequence
import numpy as np 

class NewsFetcher():
    def __init__(self,news_title,news_content,news_vert,news_subvert,news_entity):
        self.news_title = news_title
        self.news_content = news_content
        self.news_vert = news_vert
        self.news_entity = news_entity
        self.news_subvert = news_subvert
        
    def fetch(self,docids):
        bz,n = docids.shape
        news_title = self.news_title[docids] #(N,30)
        news_content = self.news_content[docids]
        news_vert = self.news_vert[docids].reshape((bz,n,1))
        news_subvert = self.news_subvert[docids].reshape((bz,n,1))
        news_entity = self.news_entity[docids]
        news_info = np.concatenate([news_title,news_vert,news_subvert,news_content,news_entity,],axis=-1)
        
        return news_info
    
    def fetch_dim1(self,docids):
        bz, = docids.shape
        news_title = self.news_title[docids] #(N,30)
        news_content = self.news_content[docids]
        news_vert = self.news_vert[docids].reshape((bz,1))
        news_subvert = self.news_subvert[docids].reshape((bz,1))
        news_entity = self.news_entity[docids]
        news_info = np.concatenate([news_title,news_vert,news_subvert,news_content,news_entity],axis=-1)
        
        return news_info


class get_hir_train_generator(Sequence):
    def __init__(self,news_fetcher,clicked_news,user_id, user_interest, entity_interest, news_id, label, batch_size):
        self.news_fetcher = news_fetcher
        self.clicked_news = clicked_news

        self.user_id = user_id
        self.doc_id = news_id
        self.label = label
        self.user_interest = np.array(user_interest)
        
        self.entity_interest = np.array(entity_interest)
        
        self.batch_size = batch_size
        self.ImpNum = self.label.shape[0]
        
        self.temp1 = self.news_fetcher.fetch(self.clicked_news[self.user_id])[:,:,30]
        self.temp2 = self.news_fetcher.fetch(self.doc_id[self.user_id])[:,:,30]
        self.temp1 = np.expand_dims(self.temp1, axis=1)
        self.temp2 = np.expand_dims(self.temp2, axis=2)
        self.interest_label = np.where(self.temp2 == self.temp1, 1, 0)#(232567, 50)
#         self.interest_label[:,1:,:] = 0
#         import pdb;pdb.set_trace()
        
        
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    

    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        label = self.label[start:ed]
        interest_label = self.interest_label[start:ed]
        
        user_interest = self.user_interest[start:ed]
        entity_interest = self.entity_interest[start:ed]
        
        doc_ids = self.doc_id[start:ed]
        info= self.news_fetcher.fetch(doc_ids)
        
        user_ids = self.user_id[start:ed]
        clicked_ids = self.clicked_news[user_ids]
        user_info = self.news_fetcher.fetch(clicked_ids)
        
        click_mask = clicked_ids>0
        click_mask = np.array(click_mask,dtype='float32')
#         return ([info, user_info],label)

        return ([info, user_info, user_interest, entity_interest],[label,interest_label])


class get_hir_user_generator(Sequence):
    def __init__(self,news_fetcher,user_interest,clicked_news,batch_size):
        self.news_fetcher = news_fetcher

        self.clicked_news = clicked_news
        
        self.user_interest = user_interest

        self.batch_size = batch_size
        self.ImpNum = self.clicked_news.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
        user_interest = self.user_interest[start:ed]
            
        clicked_ids = self.clicked_news[start:ed]
        user_info = self.news_fetcher.fetch(clicked_ids)
        
        return [user_info,user_interest,]


class get_hir_news_generator(Sequence):
    def __init__(self,news_fetcher,batch_size):
        self.news_fetcher = news_fetcher

        self.batch_size = batch_size
        self.ImpNum = news_fetcher.news_title.shape[0]
        
    def __len__(self):
        return int(np.ceil(self.ImpNum / float(self.batch_size)))

    
    def __get_news(self,docids):
        news_emb = self.news_emb[docids]

        return news_emb
    
    
    def __getitem__(self, idx):
        start = idx*self.batch_size
        ed = (idx+1)*self.batch_size
        if ed> self.ImpNum:
            ed = self.ImpNum
            
        docids = np.array([i for i in range(start,ed)])
            
        info = self.news_fetcher.fetch_dim1(docids)

        return info
